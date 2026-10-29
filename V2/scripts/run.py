#!/usr/bin/env python3
"""Clone one preserved SparkDeck TP1 deployment for opt-in source experiments."""
import argparse,copy,json,pathlib,re,secrets,subprocess,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
IMAGE='sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29'

def docker(*args):return subprocess.check_output(['docker',*args],text=True)
def inspect(name):return json.loads(docker('inspect',name))[0]
def gateway(base,deployment,action):
 req=urllib.request.Request(base.rstrip('/')+'/api/v1/deployments/'+deployment+'/'+action,data=b'{}',headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:r.read()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','rollback']);p.add_argument('--state',type=pathlib.Path,required=True);p.add_argument('--native-container');p.add_argument('--deployment-id');p.add_argument('--gateway',default='http://localhost:7878');p.add_argument('--variant',choices=['original','v1','v2'],default='v2');p.add_argument('--capture',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--fusion',choices=['single','split'],default='split');p.add_argument('--name',default='qwen38-v2-public-experiment');p.add_argument('--overlay',type=pathlib.Path);a=p.parse_args()
 if a.action=='rollback':
  state=json.loads(a.state.read_text());subprocess.run(['docker','stop',state['experiment']],check=True)
  policy=state['restart_policy'];subprocess.run(['docker','update','--restart',policy['Name'],state['native']],check=True)
  gateway(state['gateway'],state['deployment_id'],'start');return
 if not a.native_container or not a.deployment_id or not a.overlay:p.error('start requires native container, deployment ID, and generated overlay directory')
 if not re.fullmatch(r'qwen38-v2-public-[a-z0-9-]+',a.name):p.error('Use a qwen38-v2-public-* container name')
 if a.state.exists():raise RuntimeError('State file exists; preserve it for rollback')
 base=inspect(a.native_container)
 if base['Image']!=IMAGE:raise RuntimeError('Pinned runtime image differs')
 if 'CUDA_VISIBLE_DEVICES=0' not in base['Config']['Env']:raise RuntimeError('Only the preserved GPU0 deployment is supported')
 for container_id in docker('ps','-q').split():
  other=inspect(container_id)
  if other['Id'] != base['Id'] and 'CUDA_VISIBLE_DEVICES=0' in other['Config']['Env']:
   raise RuntimeError('Another GPU0 container is running; stop the prior experiment first')
 cmd=base['Config']['Cmd'];tp=cmd.index('-tp') if '-tp' in cmd else cmd.index('--tensor-parallel-size')
 if cmd[tp+1]!='1':raise RuntimeError('Only TP1 is supported')
 spec=json.loads(cmd[cmd.index('--speculative-config')+1])
 if spec.get('num_speculative_tokens')!=3:raise RuntimeError('Main experiment requires MTP3')
 cfg={k:copy.deepcopy(base['Config'][k]) for k in ['User','Env','Cmd','Image','WorkingDir','Entrypoint','Labels','ExposedPorts'] if k in base['Config']}
 cfg['Image']=IMAGE;cfg['HostConfig']=copy.deepcopy(base['HostConfig']);cfg['HostConfig']['RestartPolicy']={'Name':'no','MaximumRetryCount':0}
 if a.variant=='original':
  targets=['/vllm/models/qwen4_exp/nvidia/hyperconnection.py','/flashinfer/experimental/b12x/_lib/quant/nvfp4_csf.py','/flashinfer/experimental/b12x/_lib/quant/csf_blocks_tuning.py','/flashinfer/experimental/b12x/moe/fused_moe/_preparation.py']
  cfg['HostConfig']['Binds']=[b for b in cfg['HostConfig'].get('Binds') or [] if not any(t in b.split(':')[1] for t in targets)]
  cfg['HostConfig']['Mounts']=[m for m in cfg['HostConfig'].get('Mounts',[]) if not any(t in m['Target'] for t in targets)]
 result=a.state.parent.resolve()/'results';result.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((a.overlay/'manifest.json').read_text())
 diagnostics=isinstance(manifest,list)
 if (a.profile or a.capture) and not diagnostics:raise RuntimeError('Profiling/capture requires prepare_diagnostics output')
 if a.variant=='original' and a.capture:raise RuntimeError('Original capture requires a separately verified unmodified source overlay; not enabled by this launcher')
 entries=manifest if diagnostics else [{'source':str(a.overlay.resolve()/'gpu_worker.py'),'target':manifest['target']}]
 binds=list(cfg['HostConfig'].get('Binds') or [])
 for entry in entries:
  if entry.get('capture_only') and not a.capture:continue
  target=entry['target']
  binds=[b for b in binds if b.split(':')[1]!=target]
  cfg['HostConfig']['Mounts']=[m for m in cfg['HostConfig'].get('Mounts',[]) if m['Target']!=target]
  binds.append(entry['source']+':'+target+':ro')
 cfg['HostConfig']['Binds']=binds+[str(ROOT)+':/opt/qwen38-v2:ro',str(result)+':/opt/qwen38-v2-results:rw']
 token=secrets.token_hex(32) if diagnostics else ''
 if a.profile:
  cfg['Cmd']=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=repeat:4:sync','--output=/opt/qwen38-v2-results/'+a.name]+cfg['Cmd']+['--profiler-config','{"profiler":"cuda"}']
 oldpython=next((s.split('=',1)[1] for s in cfg['Env'] if s.startswith('PYTHONPATH=')),'')
 cfg['Env']=[s for s in cfg['Env'] if not s.startswith(('PYTHONPATH=','QWEN_V2_'))]+['PYTHONPATH=/opt/qwen38-v2:'+oldpython,'QWEN_V2_MODE='+a.variant,'QWEN_V2_CONTROL_TOKEN='+token,'QWEN_V2_CAPTURE='+str(int(a.capture)),'QWEN_V2_PROFILE='+str(int(a.profile)),'QWEN_V2_SHARED_FUSION='+(a.fusion if a.variant=='v2' else '0')]
 # Docker creates the candidate before stopping the preserved deployment.
 import http.client,socket
 class Connection(http.client.HTTPConnection):
  def connect(self):
   self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.connect('/var/run/docker.sock')
 c=Connection('localhost');c.request('POST','/v1.47/containers/create?name='+a.name,body=json.dumps(cfg),headers={'Content-Type':'application/json'});r=c.getresponse();body=r.read()
 if r.status>=300:raise RuntimeError('Container creation failed: '+body.decode())
 a.state.parent.mkdir(parents=True,exist_ok=True);a.state.touch(mode=0o600,exist_ok=False)
 a.state.write_text(json.dumps({'native':a.native_container,'experiment':a.name,'restart_policy':base['HostConfig']['RestartPolicy'],'gateway':a.gateway,'deployment_id':a.deployment_id,'control_token':token},indent=2))
 gateway(a.gateway,a.deployment_id,'stop')
 if inspect(a.native_container)['State']['Running']:raise RuntimeError('Native model did not stop; candidate remains stopped')
 subprocess.run(['docker','start',a.name],check=True)
 print('Started isolated candidate; use rollback with the same state file to restore SparkDeck')
if __name__=='__main__':main()
