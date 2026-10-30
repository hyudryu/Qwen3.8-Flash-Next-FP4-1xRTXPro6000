"""Opt-in shared FFN fusion, installed and compiled before graph preparation.

The router and routed expert path are unchanged. The custom kernel consumes
original MXFP8 weights with A16 activations; no additional weight cache is kept.
"""
import json
import os
import types
import torch
from b12x.preparation import require_prepared
from vllm.forward_context import get_forward_context
from qwen_v2.fused_shared import swiglu_projection, swiglu_projection_split, split_epilogue


DISPATCH_INSPECTION={}

def _forward(module, x):
    rows = x.shape[0]
    if (not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2
            or x.shape[1] != 2560 or rows not in module._v2_rows):
        return module._v2_reference_forward(x)
    context = get_forward_context()
    descriptor = context.batch_descriptor
    DISPATCH_INSPECTION.setdefault(str(rows), {'mode':str(context.cudagraph_runtime_mode),'descriptor':str(descriptor)})
    # The FULL graph manager deliberately sets runtime mode NONE inside its
    # outer torch.cuda.graph capture. Its uniform descriptor is authoritative.
    if descriptor is None or not descriptor.uniform:
        # Prefill, mixed batches and eager execution retain the reference path.
        return module._v2_reference_forward(x)
    try:
        prepared = require_prepared(module.down_proj.b12x_linear.plan,
                                    'gemm.blockscaled_precision', x.device)
        DISPATCH_INSPECTION[str(rows)]['prepared_mode']=str(prepared.exact[rows].config.mode)
        if prepared.exact[rows].config.mode != 'a16':
            return module._v2_reference_forward(x)
    except (KeyError, RuntimeError) as error:
        DISPATCH_INSPECTION[str(rows)]['preparation_error']=str(error)
        return module._v2_reference_forward(x)
    gate_up = module.gate_up_proj(x)[0]
    from qwen_v2.capture import shared
    shared(module, gate_up)
    gate_logits = module.expert_gate(x)[0]
    out = torch.empty((rows, 2560), dtype=x.dtype, device=x.device)
    weight = module.down_proj.b12x_mxfp8_packed_weight.weight
    if module._v2_variant == 'split':
        swiglu_projection_split[(1, 40, 8)](
            gate_up, weight.values, module._v2_scale_bytes, module._v2_scratch,
            rows, 640, 2560, 16, 64, 64, 8, out, False,
            num_warps=4, enable_fp_fusion=False)
        split_epilogue[((rows * 2560 + 511) // 512,)](
            module._v2_scratch, gate_logits, out, rows, 2560, 8, 512,
            num_warps=4, enable_fp_fusion=False)
    else:
        swiglu_projection[(1, 160)](
            gate_up, weight.values, module._v2_scale_bytes, gate_logits,
            out, out, rows, 640, 2560, 16, 16, 64, True, False,
            num_warps=4, enable_fp_fusion=False)
    return out


def install(worker):
    if os.getenv('QWEN_V2_MODE') != 'v2':
        return
    selected = os.getenv('QWEN_V2_SHARED_FUSION', '0')
    if selected not in ['1', 'single', 'split']:
        return
    if worker.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError('V2 shared fusion is restricted to TP1')
    if torch.cuda.get_device_capability(worker.device) != (12, 0):
        raise RuntimeError('V2 shared fusion is validated only on SM120')
    rows = tuple(range(1, 9))
    eligible = []
    scratch = torch.empty(8 * 8 * 2560, device=worker.device, dtype=torch.float32) if selected == 'split' else None
    for phase, model in [('target', worker.model_runner.get_model()),
                         ('draft', worker.model_runner.get_draft_model())]:
        if model is None:
            continue
        for name, module in model.named_modules():
            if type(module).__name__ != 'Qwen2MoeMLP' or module.expert_gate is None:
                continue
            packed = getattr(module.down_proj, 'b12x_mxfp8_packed_weight', None)
            if packed is None:
                continue
            weight = packed.weight
            if weight.values.dtype != torch.float8_e4m3fn:
                continue
            if tuple(weight.values.shape) != (2560, 640):
                continue
            if not weight.values.is_contiguous() or not weight.scale_rows.is_contiguous():
                continue
            if getattr(module.down_proj, 'bias', None) is not None:
                continue
            scales = weight.scale_rows.view(torch.uint8)
            if scales.numel() != 2560 * 20:
                continue
            # Setup-time validation only. Special E8M0 encodings are unsupported.
            if bool(((scales == 0) | (scales == 255)).any()):
                continue
            module._v2_scale_bytes = scales
            module._v2_reference_forward = module.forward
            module._v2_rows = rows
            module._v2_variant = selected
            module._v2_scratch = scratch
            module.forward = types.MethodType(_forward, module)
            eligible.append((phase, name, module))
    if not eligible:
        raise RuntimeError('Requested V2 shared fusion, but no compatible modules found')
    # All static row variants are compiled before vLLM preparation and capture.
    gu = torch.empty((8, 1280), device=worker.device, dtype=torch.bfloat16)
    gate = torch.empty((8, 1), device=worker.device, dtype=torch.bfloat16)
    out = torch.empty((8, 2560), device=worker.device, dtype=torch.bfloat16)
    weight = eligible[0][2].down_proj.b12x_mxfp8_packed_weight.weight
    resources = []
    for n in rows:
        if selected == 'split':
            kernel = swiglu_projection_split.warmup(
                gu, weight.values, eligible[0][2]._v2_scale_bytes, scratch,
                n, 640, 2560, 16, 64, 64, 8, out, False,
                grid=(1, 40, 8), num_warps=4, enable_fp_fusion=False)
            split_epilogue.warmup(scratch, gate, out, n, 2560, 8, 512,
                                 grid=((n * 2560 + 511) // 512,),
                                 num_warps=4, enable_fp_fusion=False)._init_handles()
        else:
            kernel = swiglu_projection.warmup(
                gu, weight.values, eligible[0][2]._v2_scale_bytes, gate, out, out,
                n, 640, 2560, 16, 16, 64, True, False,
                grid=(1, 160), num_warps=4, enable_fp_fusion=False)
        kernel._init_handles()
        from qwen_v2.resources import inspect
        resources.append({'rows': n, **inspect(kernel, worker.device)})
    torch.cuda.synchronize()
    from qwen_v2.diagnostics import OUTPUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'fusion-install.json').write_text(json.dumps({
        'modules': [{'phase': p, 'name': n} for p, n, _ in eligible],
        'variant': selected, 'eligible_rows': rows, 'resources': resources,
        'shared_scratch_bytes': 0 if scratch is None else scratch.numel() * scratch.element_size(),
        'scratch_ordering': 'Consecutive layers depend on the prior MoE join; CUDA graph replays on this single worker are serialized.',
        'persistent_added_weight_bytes': 0,
        'fallback': 'Original forward for every unsupported shape/regime',
    }, indent=2))
