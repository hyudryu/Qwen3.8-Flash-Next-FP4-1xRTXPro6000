#!/usr/bin/env python3
"""Exercise exact worker install/restore/reinstall without loading a model."""
import hashlib,pathlib,subprocess,sys,sysconfig
sys.path.insert(0,'/opt/qwen38-v2/scripts')
import install
path=pathlib.Path(sysconfig.get_paths()['purelib'])/'vllm/v1/worker/gpu_worker.py'
script='/opt/qwen38-v2/scripts/install.py'
def digest():return hashlib.sha256(path.read_bytes()).hexdigest()
def run(*args):subprocess.run([sys.executable,script,*args],check=True)
assert digest()==install.MODIFIED
run('--check');run('--restore');assert digest()==install.ORIGINAL
run();assert digest()==install.MODIFIED
run();assert digest()==install.MODIFIED
print('Worker integration roundtrip: exact original, exact reinstall, idempotent install')
