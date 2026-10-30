#!/usr/bin/env python3
"""Functional smoke cases; outputs and individual checks are retained."""
import argparse,ast,json,pathlib,re,subprocess,sys,tempfile
from serving import fetch
p=argparse.ArgumentParser();p.add_argument('--base',default='http://localhost:8001');p.add_argument('--model',default='qwen3.8-flash-next');p.add_argument('--retrieval-tokens',type=int,nargs='+',default=[16000,65536]);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();rows=[]
def chat(prompt,**options):
 body=dict(model=a.model,messages=[{'role':'user','content':prompt}],max_tokens=512,temperature=0,seed=19,chat_template_kwargs={'enable_thinking':False});body.update(options)
 return fetch(a.base,'/v1/chat/completions',body,timeout=180)
for temp,top_p in [(0,1),(0.6,0.9)]:
 try:
  r=chat('Write only Python code defining merge_intervals(intervals). Merge overlapping closed integer intervals, sort by start, return a list of two-element lists. Handle empty input. No explanation.',temperature=temp,top_p=top_p)
  text=r['choices'][0]['message'].get('content') or '';m=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S);code=m[1] if m else text
  tree=ast.parse(code)
  # The benchmark executes only a function-only answer in an isolated subprocess.
  if any(not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in tree.body):raise ValueError('Answer is not function-only Python')
  tests='\nassert merge_intervals([])==[]\nassert merge_intervals([[3,5],[1,4],[8,9]])==[[1,5],[8,9]]\nassert merge_intervals([[1,2],[2,3]])==[[1,3]]\n'
  with tempfile.TemporaryDirectory() as d:
   result=subprocess.run([sys.executable,'-I','-c',code+tests],cwd=d,capture_output=True,text=True,timeout=10)
  rows.append(dict(case='coding',temperature=temp,top_p=top_p,passed=result.returncode==0,response=r,stderr=result.stderr))
 except Exception as e:rows.append(dict(case='coding',temperature=temp,passed=False,error=str(e)))
try:
 schema={'type':'object','properties':{'ok':{'type':'boolean'},'count':{'type':'integer'}},'required':['ok','count'],'additionalProperties':False}
 r=chat('Return an object indicating success and count 7.',response_format={'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':schema}})
 value=json.loads(r['choices'][0]['message']['content']);rows.append(dict(case='structured-json',passed=value=={'ok':True,'count':7},response=r))
except Exception as e:rows.append(dict(case='structured-json',passed=False,error=str(e)))
try:
 tool={'type':'function','function':{'name':'lookup_weather','description':'Look up city weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False}}}
 r=chat('Look up weather for San Francisco.',tools=[tool],tool_choice={'type':'function','function':{'name':'lookup_weather'}})
 calls=r['choices'][0]['message'].get('tool_calls') or [];valid=bool(calls) and calls[0]['function']['name']=='lookup_weather' and 'San Francisco' in json.loads(calls[0]['function']['arguments']).get('city','');rows.append(dict(case='tool-call',passed=valid,response=r))
except Exception as e:rows.append(dict(case='tool-call',passed=False,error=str(e)))
for context_tokens in a.retrieval_tokens:
 try:
  text='Archive entry: system tests passed, cache entries expired, and the worker acknowledged the job.\n'*max(1100,context_tokens//12+100)
  ids=fetch(a.base,'/tokenize',{'model':a.model,'prompt':text,'add_special_tokens':False})['tokens'][:context_tokens]
  filler=fetch(a.base,'/detokenize',{'model':a.model,'tokens':ids})['prompt']
  mid=len(filler)//2;key='RIVER-7319-COBALT';prompt=filler[:mid]+'\nThe unique vault key is '+key+'.\n'+filler[mid:]+'\nWhat is the unique vault key? Reply with only the key.'
  r=chat(prompt,max_tokens=64);rows.append(dict(case=f'{context_tokens}-token-retrieval',passed=key in (r['choices'][0]['message'].get('content') or ''),response=r))
 except Exception as e:rows.append(dict(case=f'{context_tokens}-token-retrieval',passed=False,error=str(e)))
a.output.write_text(json.dumps({'scope':'Functional smoke, not a quality benchmark or complete correctness certification','cases':rows},indent=2));print(json.dumps([{'case':r['case'],'passed':r['passed']} for r in rows]))
