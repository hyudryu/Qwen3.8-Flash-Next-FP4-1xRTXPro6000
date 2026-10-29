#!/usr/bin/env python3
"""Cold-prefix full-vocabulary forced replay through normal target/MTP graphs."""
import argparse,json,pathlib,re,urllib.request
from serving import fetch
p=argparse.ArgumentParser();p.add_argument('--state',type=pathlib.Path,required=True);p.add_argument('--base',default='http://localhost:8001');p.add_argument('--model',default='qwen3.8-flash-next');p.add_argument('--tag',required=True);p.add_argument('--workload',type=pathlib.Path,default=pathlib.Path(__file__).parent/'workloads/forced-continuations.json');a=p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',a.tag):raise ValueError('Invalid tag')
state=json.loads(a.state.read_text());token=state.get('control_token')
if not token:raise RuntimeError('Prepare authenticated diagnostics first')
output=a.state.resolve().parent/'results';output.mkdir(exist_ok=True)
cases=json.loads(a.workload.read_text());(output/'forced-workload.json').write_text(json.dumps(cases))
def control(action,label='default'):
 req=urllib.request.Request(a.base+'/v2/control',data=json.dumps({'action':action,'label':label}).encode(),headers={'Content-Type':'application/json','X-Qwen-V2-Token':token})
 with urllib.request.urlopen(req,timeout=1800) as response:return json.load(response)
for i,case in enumerate(cases):
 if control('reset_prefix_cache') is not True:raise RuntimeError('Prefix reset failed: ensure no other traffic')
 control('bench',f'forced-start-{i}')
 try:
  result=fetch(a.base,'/v1/completions',dict(model=a.model,prompt=case['prompt'],max_tokens=len(case['continuation']),temperature=0,seed=0,ignore_eos=True))
  usage=result['usage']
  if (usage.get('prompt_tokens_details') or {}).get('cached_tokens')!=0:raise RuntimeError('Forced replay had cached prompt tokens')
  if usage['completion_tokens']!=len(case['continuation']):raise RuntimeError('Forced continuation truncated')
  (output/f'forced-{a.tag}-{i}-request.json').write_text(json.dumps({'usage':usage,'scope':'Forced diagnostic, not natural generation or throughput'},indent=2))
 finally:
  print(json.dumps(control('bench',f'forced-stop-{a.tag}-{i}')))
