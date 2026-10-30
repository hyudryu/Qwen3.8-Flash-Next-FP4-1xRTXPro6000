"""Compare a fused shared-FFN boundary on recorded decoder activations."""
import dataclasses,json,math,pathlib,statistics
import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from qwen_v2.diagnostics import OUTPUT
import importlib
from qwen_v2 import fused_shared
swiglu_projection=importlib.reload(fused_shared).swiglu_projection

def timed(fn,repeats=9):
 for _ in range(5):fn()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  for _ in range(8):fn()
 times=[]
 for _ in range(repeats):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(10):graph.replay()
  b.record();b.synchronize();times.append(a.elapsed_time(b)*1000/80)
 return {'median_us':statistics.median(times),'samples_us':times}

def run(worker):
 payload=torch.load((OUTPUT/'all-layers-activations.pt') if (OUTPUT/'all-layers-activations.pt').exists() else (OUTPUT/'mixed-c1-c8-activations.pt'),weights_only=False,map_location='cpu')['records']
 modules=dict(worker.model_runner.get_model().named_modules());results=[];all_finite=True
 with set_current_vllm_config(worker.vllm_config),set_forward_context(None,worker.vllm_config):
  for index in [0,24,47]:
   prefix=f'language_model.model.layers.{index}.mlp';module=modules[prefix+'.shared_expert'];weight=module.down_proj.b12x_mxfp8_packed_weight.weight
   print('Activation method',module.act_fn._forward_method,flush=True)
   print('Weights',index,weight.values.shape,weight.scale_rows.shape,weight.scale_rows.stride(),flush=True)
   assert weight.values.is_contiguous() and weight.scale_rows.is_contiguous()
   gu_cpu=payload['shared:target:'+prefix+'.shared_expert']['fields']['gate_up'];x_cpu=payload['moe:'+prefix+'.experts']['fields']['x']
   for rows in [1,2,4,8,16,32,64]:
    gu=gu_cpu.repeat((math.ceil(rows/gu_cpu.shape[0]),1))[:rows].to(worker.device).contiguous();x=x_cpu.repeat((math.ceil(rows/x_cpu.shape[0]),1))[:rows].to(worker.device).contiguous()
    gate_logits=module.expert_gate(x)[0].contiguous();reference_act=module.act_fn(gu);reference_projection=module.down_proj(reference_act)[0];reference=(torch.sigmoid(gate_logits)*reference_projection).clone();repeat=(torch.sigmoid(gate_logits)*module.down_proj(module.act_fn(gu))[0]).clone()
    def baseline():return torch.sigmoid(gate_logits)*module.down_proj(module.act_fn(gu))[0]
    base=timed(baseline);candidates=[]
    out=torch.empty_like(reference);actual_act=torch.empty_like(reference_act)
    for bn,bk,warps in [(16,32,4),(16,64,4),(16,128,4),(32,64,4),(64,64,4),(128,64,4),(64,128,4),(128,128,4),(64,64,8)]:
     def candidate(store=False):
      swiglu_projection[((rows+15)//16,(2560+bn-1)//bn)](gu,weight.values,weight.scale_rows.view(torch.uint8),gate_logits,out,actual_act,rows,640,2560,16,bn,bk,True,store,num_warps=warps,enable_fp_fusion=False)
      return out
     candidate(True);torch.cuda.synchronize()
     diff=(reference.float()-out.float()).abs();actdiff=(reference_act.float()-actual_act.float()).abs();finite=bool(torch.isfinite(out).all());all_finite &= finite
     row={'bn':bn,'bk':bk,'warps':warps,'max_abs':float(diff.max()),'rms':float(diff.square().mean().sqrt()),'activation_max_abs':float(actdiff.max()),'activation_exact_fraction':float((reference_act==actual_act).float().mean()),'finite':finite,**timed(candidate)};candidates.append(row)
    results.append({'layer':index,'rows':rows,'recorded_input_rows':gu_cpu.shape[0],'reference':base,'reference_repeat_max_abs':float((reference-repeat).float().abs().max()),'candidates':candidates})
    print('Shared FFN layer',index,'rows',rows,'original_us',base['median_us'],'best_us',min(r['median_us'] for r in candidates),'max_abs',max(r['max_abs'] for r in candidates),flush=True)
 path=OUTPUT/'shared-fusion-micro.json';path.write_text(json.dumps({'scope':'Recorded real gate/up activations and inputs from the loaded checkpoint. Rows are packed/repeated for controlled geometry coverage; not all shapes claimed observed in production. Only activation→down-projection→shared-gate boundary timed; input projection and gate-logit projection excluded identically.','all_finite':all_finite,'records':results},indent=2));return {'path':str(path),'cases':len(results),'all_finite':all_finite}
