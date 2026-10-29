#!/usr/bin/env python3
"""Strict, hash-checked unified patches for the pinned serving image."""
import argparse,hashlib,json,pathlib,re,shutil,sysconfig
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def apply(original,patch):
 lines=original.decode().splitlines(keepends=True);diff=patch.splitlines(keepends=True);out=[];cursor=0;i=2
 while i<len(diff):
  m=re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@',diff[i]);assert m,'Unexpected patch header'
  start=int(m[1])-1;assert start>=cursor
  out.extend(lines[cursor:start]);cursor=start;i+=1
  while i<len(diff) and not diff[i].startswith('@@ '):
   line=diff[i];i+=1
   if line.startswith((' ','-')):
    assert cursor<len(lines) and lines[cursor]==line[1:],'Patch context mismatch'
    if line.startswith(' '):out.append(lines[cursor])
    cursor+=1
   elif line.startswith('+'):out.append(line[1:])
   else:raise ValueError('Unsupported patch line')
 out.extend(lines[cursor:]);return ''.join(out).encode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--site-packages',type=pathlib.Path,default=pathlib.Path(sysconfig.get_paths()['purelib']));p.add_argument('--backup',type=pathlib.Path,default=ROOT/'backups');p.add_argument('--restore',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
 manifest=json.loads((ROOT/'patches/manifest.json').read_text());prepared=[]
 # Validate every target before making any mutation.
 for item in manifest:
  target=a.site_packages/item['target'];current=target.read_bytes();digest=sha(current);backup=a.backup/item['target']
  if a.restore:
   assert digest in (item['original_sha256'],item['modified_sha256']),str(target)
   data=backup.read_bytes() if digest==item['modified_sha256'] else current
   assert sha(data)==item['original_sha256'];prepared.append((target,data,None));continue
  assert digest in (item['original_sha256'],item['modified_sha256']),f'Unsupported upstream source: {target}; use the pinned image'
  data=apply(current,(ROOT/'patches'/item['patch']).read_text()) if digest==item['original_sha256'] else current
  assert sha(data)==item['modified_sha256'];prepared.append((target,data,current if digest==item['original_sha256'] else None))
 if not a.check:
  for (target,data,original),item in zip(prepared,manifest):
   if original is not None:
    backup=a.backup/item['target'];backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.exists():assert sha(backup.read_bytes())==item['original_sha256']
    else:backup.write_bytes(original)
   target.write_bytes(data)
 print(('Validated' if a.check else 'Restored' if a.restore else 'Installed')+' all four V1 inference source patches')
if __name__=='__main__':main()
