#!/usr/bin/env python3
"""Authenticated diagnostics for an explicitly prepared experiment worker."""
import argparse,json,pathlib,urllib.request
p=argparse.ArgumentParser();p.add_argument('--state',type=pathlib.Path,required=True);p.add_argument('--base',default='http://localhost:8001');p.add_argument('action');p.add_argument('label',nargs='?',default='default');a=p.parse_args()
token=json.loads(a.state.read_text()).get('control_token')
if not token:raise RuntimeError('This experiment was not prepared for diagnostics')
req=urllib.request.Request(a.base+'/v2/control',data=json.dumps({'action':a.action,'label':a.label}).encode(),headers={'Content-Type':'application/json','X-Qwen-V2-Token':token})
with urllib.request.urlopen(req,timeout=1800) as r:print(r.read().decode())
