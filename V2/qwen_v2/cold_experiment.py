"""Compare hot replay and explicitly evicted-weight shared FFN boundaries."""
import importlib,json,math,statistics
import torch,triton
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from qwen_v2 import fused_shared
from qwen_v2.diagnostics import OUTPUT

def run(worker):
 f=importlib.reload(fused_shared)
 payload=torch.load(OUTPUT/'all-layers-activations.pt',map_location='cpu',weights_only=False)['records'];model=worker.model_runner.get_model();module=dict(model.named_modules())['language_model.model.layers.24.mlp.shared_expert'];w=module.down_proj.b12x_mxfp8_packed_weight.weight
 gu_cpu=payload['shared:target:language_model.model.layers.24.mlp.shared_expert']['fields']['gate_up'];x_cpu=payload['moe:language_model.model.layers.24.mlp.experts']['fields']['x']
 evict=torch.empty(256*1024*1024,device=worker.device,dtype=torch.uint8);results=[]
 def timed(fn,cold):
  for _ in range(3):fn()
  samples=[]
  for _ in range(15):
   if cold:evict.zero_()
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000)
  return {'median_us':statistics.median(samples),'samples_us':samples}
 with set_current_vllm_config(worker.vllm_config),set_forward_context(None,worker.vllm_config):
  for n in [1,4,8,32]:
   gu=gu_cpu.repeat((math.ceil(n/len(gu_cpu)),1))[:n].to(worker.device).contiguous();x=x_cpu.repeat((math.ceil(n/len(x_cpu)),1))[:n].to(worker.device).contiguous();gate=module.expert_gate(x)[0];out=torch.empty(n,2560,device=worker.device,dtype=torch.bfloat16)
   def reference():return torch.sigmoid(gate)*module.down_proj(module.act_fn(gu))[0]
   ref=reference().clone();base={'hot':timed(reference,False),'cold':timed(reference,True)}
   configs=[]
   for bn,bk,splits in [(bn,bk,s) for bn in [16,32,64] for bk in [32,64,128] for s in [2,4,8]]:
    part=torch.empty(splits*n*2560,device=worker.device,dtype=torch.float32)
    def candidate():
     f.swiglu_projection_split[(triton.cdiv(n,16),triton.cdiv(2560,bn),splits)](gu,w.values,w.scale_rows.view(torch.uint8),part,n,640,2560,16,bn,bk,splits,out,False,num_warps=4,enable_fp_fusion=False)
     f.split_epilogue[(triton.cdiv(n*2560,512),)](part,gate,out,n,2560,splits,512,num_warps=4,enable_fp_fusion=False)
    candidate();delta=(ref.float()-out.float()).abs();configs.append(dict(bn=bn,bk=bk,splits=splits,max_abs=float(delta.max()),rms=float(delta.square().mean().sqrt()),hot=timed(candidate,False),cold=timed(candidate,True)))
   results.append(dict(rows=n,reference=base,candidates=configs));print('Cold shared',n,'reference',base['cold']['median_us'],'best',min(r['cold']['median_us'] for r in configs),flush=True)
 path=OUTPUT/'shared-split-cold-micro.json';path.write_text(json.dumps({'scope':'Recorded activations/weights. Cold timing explicitly zeroes a 256MiB eviction buffer before the timed boundary. This diagnostic is not an inference trace or end-to-end result. Hot timing here uses CUDA events with host gaps, unlike the tight graph replay microbenchmark.','records':results},indent=2));return {'path':str(path),'cases':len(results)}
