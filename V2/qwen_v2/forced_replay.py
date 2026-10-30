"""Explicit C1 teacher-forcing diagnostic; never installed in serving mode.

The normal target and MTP graphs execute. Sampling is overridden only for this
accuracy diagnostic, committing one forced token and rejecting all draft rows.
Raw full-vocabulary logits are copied to GPU storage before sampling modifies
anything. Host copies happen only after the diagnostic request completes.
"""
import json,types
import torch
from qwen_v2.diagnostics import OUTPUT
STATE=None

def start(worker,label):
 global STATE
 if STATE is not None:raise RuntimeError('A forced diagnostic is already active')
 case=int(label.rsplit('-',1)[1]);data=json.loads((OUTPUT/'forced-workload.json').read_text())[case]
 runner=worker.model_runner;model=runner.get_model();trace=data['continuation']
 vocab=worker.model_runner.vocab_size
 state=dict(step=0,trace=trace,logits=torch.empty((len(trace),vocab),device=worker.device,dtype=torch.float32),model=model,runner=runner,original_logits=model.compute_logits,original_sample=runner.sample,source_dtypes=set())
 def logits_method(this,hidden,*args,**kwargs):
  value=state['original_logits'](hidden,*args,**kwargs)
  if state['step']<len(trace):
   state['logits'][state['step']].copy_(value[0]);state['source_dtypes'].add(str(value.dtype))
  return value
 def sample_method(this,hidden,batch,grammar):
  if batch.num_reqs!=1:raise RuntimeError('Forced replay requires isolated C1')
  result=state['original_sample'](hidden,batch,grammar)
  i=state['step']
  if i>=len(trace):raise RuntimeError('Forced continuation exhausted')
  output=result[0];output.sampled_token_ids.fill_(-1);output.sampled_token_ids[0,0].fill_(trace[i]);output.num_sampled.fill_(1)
  rejected=int(batch.cu_num_logits_np[1]-batch.cu_num_logits_np[0])-1
  output.num_rejected.fill_(rejected);state['step']+=1
  return output,output.num_sampled,output.num_rejected
 model.compute_logits=types.MethodType(logits_method,model);runner.sample=types.MethodType(sample_method,runner);STATE=state
 return {'case':case,'tokens':len(trace),'diagnostic_gpu_bytes':state['logits'].numel()*4}

def stop(worker,label):
 global STATE
 state=STATE
 if state is None:raise RuntimeError('No forced diagnostic active')
 state['model'].compute_logits=state['original_logits'];state['runner'].sample=state['original_sample']
 torch.cuda.synchronize();n=state['step'];path=OUTPUT/(label+'-logits.pt')
 torch.save({'logits':state['logits'][:n].cpu(),'tokens':state['trace'][:n],'source_dtypes':sorted(state['source_dtypes']),'scope':'Raw full-vocabulary target logits, C1 forced one-token commit with MTP3 executed and remaining draft tokens rejected; diagnostic only.'},path)
 STATE=None
 return {'path':str(path),'tokens':n,'source_dtypes':sorted(state['source_dtypes'])}
