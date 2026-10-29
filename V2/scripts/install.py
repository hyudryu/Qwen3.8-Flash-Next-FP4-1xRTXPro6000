#!/usr/bin/env python3
"""Hash-checked, reversible worker integration for the pinned runtime."""
import argparse,hashlib,pathlib,sysconfig
ORIGINAL='366c271edbfc8fafecb3eb0756058e9684f28c1e048484f32a2330560afb2bd4'
MODIFIED='d91a6fcb6ce2243d3f2e190e3b1dab209edd9de3a946a088717fad052bc542ca'
NEEDLE='            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)'
ADDITION='\n            from qwen_v2.hooks import install\n            install(self)'

def main():
 p=argparse.ArgumentParser();p.add_argument('--site-packages',type=pathlib.Path,default=pathlib.Path(sysconfig.get_paths()['purelib']));p.add_argument('--backup',type=pathlib.Path,default=pathlib.Path('/opt/qwen38-v2/backups/gpu_worker.py'));p.add_argument('--restore',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
 target=a.site_packages/'vllm/v1/worker/gpu_worker.py';data=target.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if digest not in [ORIGINAL,MODIFIED]:raise RuntimeError('Unsupported worker source; use the pinned runtime')
 if a.restore:
  updated=a.backup.read_bytes() if digest==MODIFIED else data
  if hashlib.sha256(updated).hexdigest()!=ORIGINAL:raise RuntimeError('Backup hash differs')
 else:
  text=data.decode()
  if digest==ORIGINAL:
   if text.count(NEEDLE)!=1:raise RuntimeError('Initialization anchor differs')
   updated=text.replace(NEEDLE,NEEDLE+ADDITION).encode()
  else:updated=data
  if hashlib.sha256(updated).hexdigest()!=MODIFIED:raise RuntimeError('Patched worker hash differs')
 if not a.check:
  if digest==ORIGINAL and not a.restore:
   a.backup.parent.mkdir(parents=True,exist_ok=True)
   if a.backup.exists() and hashlib.sha256(a.backup.read_bytes()).hexdigest()!=ORIGINAL:raise RuntimeError('Existing backup differs')
   if not a.backup.exists():a.backup.write_bytes(data)
  target.write_bytes(updated)
 print('Validated' if a.check else 'Restored' if a.restore else 'Installed', 'V2 worker integration')
if __name__=='__main__':main()
