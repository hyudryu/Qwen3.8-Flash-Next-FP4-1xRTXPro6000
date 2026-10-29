"""Exercise install/check/restore on a disposable V1 image container."""
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/opt/qwen38-v1');script=str(root/'scripts/install.py');manifest=json.loads((root/'patches/manifest.json').read_text());site=pathlib.Path('/opt/venv/lib/python3.12/site-packages')
for action,key in [('--check','modified_sha256'),('--restore','original_sha256'),(None,'modified_sha256')]:
 subprocess.run([sys.executable,script]+([action] if action else []),check=True)
 for item in manifest:assert hashlib.sha256((site/item['target']).read_bytes()).hexdigest()==item[key]
print('All four sources pass exact install/restore/reinstall SHA256 roundtrip')
