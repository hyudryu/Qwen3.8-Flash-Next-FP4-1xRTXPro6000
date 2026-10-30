#!/usr/bin/env python3
"""Generate a worker overlay from a preserved pinned container; retain its notices."""
import argparse,hashlib,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--container',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
source='/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py'
a.output.mkdir(parents=True,exist_ok=False)
subprocess.run(['docker','cp',a.container+':'+source,str(a.output/'gpu_worker.original.py')],check=True)
original=(a.output/'gpu_worker.original.py').read_text()
from install import ORIGINAL,MODIFIED
if hashlib.sha256(original.encode()).hexdigest()!=ORIGINAL:raise RuntimeError('Worker hash differs from pinned runtime')
needle='            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)'
if original.count(needle)!=1:raise RuntimeError('Unsupported worker source: initialization anchor differs')
modified=original.replace(needle,needle+'\n            from qwen_v2.hooks import install\n            install(self)')
if hashlib.sha256(modified.encode()).hexdigest()!=MODIFIED:raise RuntimeError('Modified worker hash differs')
compile(modified,'gpu_worker.py','exec');(a.output/'gpu_worker.py').write_text(modified)
(a.output/'manifest.json').write_text(json.dumps({'target':source,'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'modified_sha256':hashlib.sha256(modified.encode()).hexdigest()},indent=2))
print('Prepared opt-in worker overlay; model launch settings unchanged')
