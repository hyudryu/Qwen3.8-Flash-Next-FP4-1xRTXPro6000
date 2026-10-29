"""Opt-in bounded real-activation snapshots; absent in performance variants."""
import json,os,pathlib,re
import torch,triton
import triton.language as tl
ENABLED=os.getenv('QWEN_V2_CAPTURE')=='1';FLAG=None;BUFFERS={};MODULE_KEYS={}
@triton.jit(do_not_specialize=['N'])
def _copy_matrix(SRC,DST,META,FLAG,N,W:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
 if (tl.load(FLAG) != 0) & (N >= tl.load(META)):
  i=tl.program_id(0)*B+tl.arange(0,B);r=i//W;c=i%W
  value=tl.load(SRC+r*S+c,i<N*W,other=0)
  tl.store(DST+i,value,i<N*W)
  if tl.program_id(0)==0:
   tl.store(META,N);tl.atomic_add(META+1,1)
def _entry(key,fields,device):
 BUFFERS[key]={'fields':{name:torch.zeros((64,width),device=device,dtype=dtype) for name,width,dtype in fields},'meta':torch.zeros(2,device=device,dtype=torch.int32)}
def initialize(worker):
 global FLAG
 if not ENABLED:return
 FLAG=torch.zeros(1,device=worker.device,dtype=torch.int32)
 models=[('target',worker.model_runner.get_model()),('draft',worker.model_runner.get_draft_model())]
 for phase,model in models:
  if model is None:continue
  for name,module in model.named_modules():
   match=re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)',name)
   if not match:continue
   if phase=='target' and int(match[1]) not in [0,24,47] and type(module).__name__ not in ['MoERunner','Qwen2MoeMLP']:continue
   typ=type(module).__name__;key=phase+':'+name;MODULE_KEYS[id(module)]=key
   if typ=='GatedResidual':
    _entry('hc:'+key,[('normalized',10240,torch.bfloat16)],worker.device)
    _entry('combine:'+key,[('hidden',10240,torch.bfloat16),('block',2560,torch.bfloat16),('injection',4,torch.bfloat16)],worker.device)
   elif typ=='MoERunner':
    _entry('moe:'+module.layer_name,[('x',2560,torch.bfloat16),('ids',10,torch.int32),('weights',10,torch.float32)],worker.device)
   elif typ=='Qwen2MoeMLP':_entry('shared:'+key,[('gate_up',1280,torch.bfloat16)],worker.device)
 # Explicit preparation of the copy geometries, before any graph capture.
 for entry in BUFFERS.values():
  for name,dst in entry['fields'].items():
   for stride in ([4,336] if dst.shape[1]==4 else [dst.shape[1]]):
    _copy_matrix.warmup(dst.dtype,dst.dtype,torch.int32,torch.int32,4,W=dst.shape[1],S=stride,B=512,grid=(1,))
def store(key,values):
 if not ENABLED or key not in BUFFERS:return
 entry=BUFFERS[key]
 for name,value in values.items():
  if value is None or not 0<value.shape[0]<=64 or value.ndim!=2:continue
  dst=entry['fields'][name]
  if value.dtype!=dst.dtype:
   if name=='ids' and value.dtype==torch.int64:continue
   raise ValueError((name,value.dtype,dst.dtype))
  _copy_matrix[(triton.cdiv(value.shape[0]*value.shape[1],512),)](value,dst,entry['meta'],FLAG,value.shape[0],value.shape[1],value.stride(0),512)
def hc(module,normalized):
 if ENABLED:store('hc:'+MODULE_KEYS.get(id(module),''),{'normalized':normalized})
def combine(module,hidden,block,injection):
 if ENABLED:store('combine:'+MODULE_KEYS.get(id(module),''),{'hidden':hidden,'block':block,'injection':injection})
def moe(module,x,ids,weights):
 if ENABLED:store('moe:'+module.layer_name,{'x':x,'ids':ids,'weights':weights})
def shared(module,gate_up):
 if ENABLED:store('shared:'+MODULE_KEYS.get(id(module),''),{'gate_up':gate_up})
def start():
 if not ENABLED:raise RuntimeError('Activation capture not enabled')
 FLAG.zero_();torch.cuda.synchronize()
 for entry in BUFFERS.values():entry['meta'].zero_()
 FLAG.fill_(1);torch.cuda.synchronize()
def stop(label,output):
 FLAG.zero_();torch.cuda.synchronize();data={}
 for key,entry in BUFFERS.items():
  n,calls=entry['meta'].cpu().tolist()
  if n:data[key]={'rows':n,'field_copy_calls':calls,'fields':{name:t[:n].cpu() for name,t in entry['fields'].items()}}
 path=output/(label+'-activations.pt');torch.save({'scope':'Largest-row real invocation per selected layer/boundary; mixed padded rows retained. Capture timings are not serving performance.','records':data},path)
 return {'path':str(path),'boundaries':len(data),'allocated_capture_bytes':sum(t.numel()*t.element_size() for e in BUFFERS.values() for t in e['fields'].values())}
