"""All-layer replay validation on bounded snapshots from the loaded model."""
import json, math
import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
import importlib
from qwen_v2 import fused_shared
from qwen_v2.diagnostics import OUTPUT

def run(worker,split=False):
    f=importlib.reload(fused_shared)
    payload=torch.load(OUTPUT/'all-layers-activations.pt',map_location='cpu',weights_only=False)['records']
    records=[]
    with set_current_vllm_config(worker.vllm_config),set_forward_context(None,worker.vllm_config):
      for phase,model in [('target',worker.model_runner.get_model()),('draft',worker.model_runner.get_draft_model())]:
        if model is None:continue
        modules=dict(model.named_modules())
        for name,module in modules.items():
          key='shared:'+phase+':'+name
          if key not in payload:continue
          moe_name=name.rsplit('.shared_expert',1)[0]+'.experts'
          moe=modules[moe_name]
          x_cpu=payload['moe:'+moe.layer_name]['fields']['x'];gu_cpu=payload[key]['fields']['gate_up']
          if not hasattr(module.down_proj,'b12x_mxfp8_packed_weight'):continue
          w=module.down_proj.b12x_mxfp8_packed_weight.weight
          for n in [1,2,4,8,16,32,64]:
            x=x_cpu.repeat((math.ceil(n/len(x_cpu)),1))[:n].to(worker.device).contiguous()
            gu=gu_cpu.repeat((math.ceil(n/len(gu_cpu)),1))[:n].to(worker.device).contiguous()
            gate=module.expert_gate(x)[0];act=module.act_fn(gu)
            ref=torch.sigmoid(gate)*module.down_proj(act)[0]
            out=torch.empty_like(ref);intermediate=torch.empty_like(act)
            if split:
                part=torch.empty(8*n*2560,device=worker.device,dtype=torch.float32)
                f.swiglu_projection_split[((n+15)//16,40,8)](gu,w.values,w.scale_rows.view(torch.uint8),part,n,640,2560,16,64,64,8,intermediate,True,num_warps=4,enable_fp_fusion=False)
                f.split_epilogue[((n*2560+511)//512,)](part,gate,out,n,2560,8,512,num_warps=4,enable_fp_fusion=False)
            else:
                f.swiglu_projection[((n+15)//16,160)](gu,w.values,w.scale_rows.view(torch.uint8),gate,out,intermediate,n,640,2560,16,16,64,True,True,num_warps=4,enable_fp_fusion=False)
            delta=(out.float()-ref.float()).abs();ad=(act.float()-intermediate.float()).abs()
            records.append(dict(phase=phase,name=name,rows=n,recorded_rows=len(gu_cpu),activation_max_abs=float(ad.max()),activation_equal_fraction=float((act==intermediate).float().mean()),output_max_abs=float(delta.max()),output_rms=float(delta.square().mean().sqrt()),output_equal_fraction=float((out==ref).float().mean()),finite=bool(torch.isfinite(out).all())))
    p=OUTPUT/('all-layer-shared-split-correctness.json' if split else 'all-layer-shared-correctness.json');p.write_text(json.dumps({'scope':'Real recorded activations/weights, repeated rows for geometry coverage. This is intermediate replay validation, not full-model distribution validation.','records':records},indent=2));return {'path':str(p),'cases':len(records),'activation_max_abs':max(r['activation_max_abs'] for r in records),'output_max_abs':max(r['output_max_abs'] for r in records)}
