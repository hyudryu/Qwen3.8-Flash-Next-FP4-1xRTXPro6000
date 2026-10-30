"""Idle-worker activation replay harness and preparation inspection."""
import json,pathlib,torch
from qwen_v2.diagnostics import OUTPUT,summarize

def run(worker,label):
 if label.startswith('source-hashes-'):
  from qwen_v2.source_hashes import run as hashes
  return hashes(label)
 if label=='dispatch':
  from qwen_v2.hooks import DISPATCH_INSPECTION
  return DISPATCH_INSPECTION
 if label.startswith('forced-start-'):
  from qwen_v2.forced_replay import start
  return start(worker,label)
 if label.startswith('forced-stop-'):
  from qwen_v2.forced_replay import stop
  return stop(worker,label)
 if label=='shared-cold':
  import importlib
  from qwen_v2 import cold_experiment
  return importlib.reload(cold_experiment).run(worker)
 if label in ['shared-validation','shared-validation-split']:
  import importlib
  from qwen_v2 import validate_shared
  return importlib.reload(validate_shared).run(worker,split=label.endswith('-split'))
 if label=='shared-fusion':
  import importlib
  from qwen_v2 import shared_experiment
  return importlib.reload(shared_experiment).run(worker)
 if label!='inventory':raise ValueError(label)
 model=worker.model_runner.get_model();layers=[m for m in model.modules() if type(m).__name__=='Qwen4ExpDecoderLayer'];layer=layers[24];shared=layer.mlp.shared_expert
 data={}
 for name,obj in [('shared_up',shared.gate_up_proj.b12x_linear),('shared_down',shared.down_proj.b12x_linear),('up_weight',shared.gate_up_proj.b12x_mxfp8_packed_weight),('down_weight',shared.down_proj.b12x_mxfp8_packed_weight),('experts',layer.mlp.experts.routed_experts.quant_method.backend),('mlp_hc',layer.mlp_hyper_connection)]:
  data[name]={'type':type(obj).__name__,'fields':{k:summarize(v) for k,v in vars(obj).items()}}
 from b12x.preparation import require_prepared
 for name,holder in [('shared_up',shared.gate_up_proj.b12x_linear),('shared_down',shared.down_proj.b12x_linear)]:
  state=require_prepared(holder.plan,'gemm.blockscaled_precision',worker.device)
  data[name]['prepared_type']=type(state).__name__
  data[name]['prepared_fields']=list(vars(state))
  data[name]['regimes']={str(n):{'query':str(st.query),'config':str(st.config),'offsets':summarize(st.offsets),'workspace':st.required_workspace,'programs':list(st.programs)} for n,st in state.exact.items()}
 path=OUTPUT/'prepared-inventory.json' ;path.write_text(json.dumps(data,indent=2));return {'path':str(path)}
