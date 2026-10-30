"""Explicit idle-worker diagnostics; no serving-time host tensor readbacks."""
import collections,json,os,pathlib,re,time
import torch
PROFILE=os.getenv('QWEN_V2_PROFILE')=='1'
WORKER=None;ACTIVE=PROFILE;SCHEDULE=[];MODULES={};OUTPUT=pathlib.Path('/opt/qwen38-v2-results')
def schedule(phase,actual,padded,kind='forward',requests=None):
 if ACTIVE:
  if PROFILE:torch.cuda.nvtx.mark(f'qwen_v2|{phase}|{kind}|{actual}|{padded}|{requests}')
  SCHEDULE.append({'host_time_ns':time.perf_counter_ns(),'phase':phase,'actual_rows':int(actual),'padded_rows':int(padded),'kind':kind,'requests':None if requests is None else int(requests)})
def summarize(value,depth=0):
 if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'dtype':str(value.dtype),'stride':list(value.stride()),'bytes':value.numel()*value.element_size()}
 if isinstance(value,(int,float,str,bool)) or value is None:return value
 if depth>1:return type(value).__name__
 if isinstance(value,(tuple,list)):return [summarize(x,depth+1) for x in value]
 if isinstance(value,dict):return {str(k):summarize(v,depth+1) for k,v in value.items()}
 return type(value).__name__
def initialize(worker):
 global WORKER
 WORKER=worker;OUTPUT.mkdir(parents=True,exist_ok=True)
 target=worker.model_runner.get_model();draft=worker.model_runner.get_draft_model();rows=[]
 for phase,model in [('target',target),('draft',draft)]:
  if model is None:continue
  for name,module in model.named_modules():
   typ=type(module).__name__
   if typ in ['Qwen4ExpDecoderLayer','GatedResidual','Qwen4ExpSparseMoeBlock','Qwen2MoeMLP','MoERunner'] or ('layers.24.' in name and any(part in name for part in ['gate','proj'])):
    MODULES[phase+':'+name]=module
    row={'phase':phase,'name':name,'type':typ,'fields':{k:summarize(v) for k,v in vars(module).items() if not k.startswith('_')},'parameters':{n:summarize(p) for n,p in module.named_parameters(recurse=False)}}
    row['children']={n:type(m).__name__ for n,m in module.named_children()};rows.append(row)
 (OUTPUT/'module-inventory.json').write_text(json.dumps(rows,indent=2))
 from qwen_v2 import capture,hooks
 capture.initialize(worker)
 hooks.install(worker)
def control(worker,action,label='default'):
 global ACTIVE
 if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',label):raise ValueError('Invalid diagnostic label')
 if action=='start':SCHEDULE.clear();ACTIVE=True;return {'started':True}
 if action=='stop':
  ACTIVE=False;counts=collections.Counter((r['phase'],r['kind'],r['actual_rows'],r['padded_rows']) for r in SCHEDULE)
  out={'scope':'Host dispatch arguments at actual graph launches, not inferred from concurrency. No tensor host readbacks.','records':SCHEDULE,'histograms':[{'phase':k[0],'kind':k[1],'actual_rows':k[2],'padded_rows':k[3],'count':v} for k,v in sorted(counts.items())]}
  path=OUTPUT/(label+'-rows.json');path.write_text(json.dumps(out,indent=2));return {'path':str(path),'calls':len(SCHEDULE)}
 if action=='capture_start':
  from qwen_v2 import capture
  capture.start();return {'started':True}
 if action=='capture_stop':
  from qwen_v2 import capture
  return capture.stop(label,OUTPUT)
 if action=='inspect':return {'path':str(OUTPUT/'module-inventory.json'),'modules':len(MODULES)}
 if action=='bench':
  import importlib
  module=importlib.import_module('qwen_v2.worker_experiments')
  return importlib.reload(module).run(worker,label)
 raise ValueError('Unsupported diagnostic action')
