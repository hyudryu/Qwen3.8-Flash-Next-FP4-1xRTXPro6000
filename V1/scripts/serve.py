#!/usr/bin/env python3
"""Run one TP1 V1 instance without altering any existing container."""
import argparse,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--image',default='qwen38-fp4-tp1:v1');p.add_argument('--name',default='qwen38-v1');p.add_argument('--port',type=int,default=8000);p.add_argument('--hf-cache',type=pathlib.Path,default=pathlib.Path.home()/'.cache/huggingface');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
args=json.loads((ROOT/'scripts/serve-args.json').read_text())
cmd=['docker','run','-d','--name',a.name,'--restart','unless-stopped','--gpus','device=0','--ipc=host','--shm-size','16g','--env-file',str(ROOT/'scripts/serve.env'),'-p',f'{a.port}:8000','-v',str(a.hf_cache.resolve())+':/root/.cache/huggingface:rw','--entrypoint','/opt/venv/bin/vllm',a.image,'serve','nvidia/Qwen3.8-Flash-Next-FP4-Local',*args]
if a.dry_run:print(shlex.join(cmd))
else:subprocess.run(cmd,check=True)
