#!/usr/bin/env python3
"""Materialize diagnostics outside V1; upstream headers remain intact."""
import argparse,hashlib,json,pathlib,subprocess,tempfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--container',required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
ORIGINAL_HASHES={}
def read(relative):
    with tempfile.TemporaryDirectory() as directory:
        path=pathlib.Path(directory)/'source.py'
        subprocess.run(['docker','cp',a.container+':/opt/venv/lib/python3.12/site-packages/'+relative,str(path)],check=True)
        data=path.read_text();ORIGINAL_HASHES[relative]=hashlib.sha256(data.encode()).hexdigest()
        return data
files={}
worker=read('vllm/v1/worker/gpu_worker.py');needle='            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)';assert worker.count(needle)==1
worker=worker.replace(needle,needle+'\n            from qwen_v2.diagnostics import initialize\n            initialize(self)')
worker=worker.replace('    def update_config(self, overrides:', '    def qwen_v2_control(self, action, label="default"):\n        from qwen_v2.diagnostics import control\n        return control(self, action, label)\n\n    def update_config(self, overrides:',1)
files['v1/worker/gpu_worker.py']=worker
api=read('vllm/entrypoints/launchers/api_server/routers.py');needle='    register_vllm_serve_api_routers(app)';assert api.count(needle)==1
api=api.replace(needle,needle+'''
    import os,secrets
    from fastapi import Request,HTTPException
    @app.post('/v2/control')
    async def qwen_v2_control(request: Request):
        token=os.environ.get('QWEN_V2_CONTROL_TOKEN','')
        if not token or not secrets.compare_digest(request.headers.get('X-Qwen-V2-Token',''),token):
            raise HTTPException(403,'Diagnostic token required')
        body=await request.json()
        if body['action']=='reset_prefix_cache':
            return await request.app.state.engine_client.reset_prefix_cache(reset_running_requests=False,reset_connector=False)
        return await request.app.state.engine_client.collective_rpc('qwen_v2_control',args=(body['action'],body.get('label','default')))
''');files['entrypoints/launchers/api_server/routers.py']=api
runner=read('vllm/v1/worker/gpu/model_runner.py');needle='        # Run model.\n        if batch_desc.cg_mode == CUDAGraphMode.FULL:';assert runner.count(needle)==1
runner=runner.replace(needle,"        from qwen_v2.diagnostics import schedule\n        schedule('target',input_batch.num_tokens,input_batch.num_tokens_after_padding,'target-forward',input_batch.num_reqs)\n\n"+needle);files['v1/worker/gpu/model_runner.py']=runner
spec=read('vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py')
# Use runtime host arguments outside graph replay for true scheduled shapes.
needle='            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)';assert spec.count(needle)==1
spec=spec.replace(needle,"            from qwen_v2.diagnostics import schedule\n            schedule('draft',input_batch.num_tokens,prefill_batch_desc.num_tokens,'draft-prefill',num_reqs)\n"+needle)
needle='                self.decode_cudagraph_manager.run_fullgraph(batch_desc)';assert spec.count(needle)==1
spec=spec.replace(needle,"                from qwen_v2.diagnostics import schedule\n                schedule('draft',num_reqs,batch_desc.num_tokens,'draft-step',num_reqs)\n"+needle)
needle='            self.decode_cudagraph_manager.run_fullgraph(batch_desc)';# distinct fused indentation after previous replace leaves substring too; replace by complete line only
lines=spec.splitlines();out=[]
for line in lines:
 if line==needle:
  out.extend(["            from qwen_v2.diagnostics import schedule","            schedule('draft',num_reqs,batch_desc.num_tokens,'draft-fused-steps',num_reqs)"])
 out.append(line)
spec='\n'.join(out)+'\n';files['v1/worker/gpu/spec_decode/autoregressive/speculator.py']=spec
capture_files={}
moe=read('vllm/model_executor/layers/fused_moe/runner/moe_runner.py')
needle='            if workspace is None:\n                fused_out = self.routed_experts.forward_modular('
assert moe.count(needle)==1
moe=moe.replace(needle,"            from qwen_v2.capture import moe as capture_moe\n            capture_moe(self,hidden_states,topk_ids,topk_weights)\n\n"+needle)
capture_files['vllm/model_executor/layers/fused_moe/runner/moe_runner.py']=moe
hc=read('vllm/models/qwen4_exp/nvidia/hyperconnection.py')
needle='    def _mix_normalized(self, normalized: torch.Tensor):'
hc=hc.replace(needle,needle+"\n        from qwen_v2.capture import hc as capture_hc\n        capture_hc(self,normalized)")
needle='    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):'
hc=hc.replace(needle,needle+"\n        from qwen_v2.capture import combine as capture_combine\n        if prev_block_output is not None: capture_combine(self,hidden_states,prev_block_output,prev_injection)")
capture_files['vllm/models/qwen4_exp/nvidia/hyperconnection.py']=hc
shared=read('vllm/model_executor/models/qwen2_moe.py');needle='        gate_up, _ = self.gate_up_proj(x)'
shared=shared.replace(needle,needle+"\n        from qwen_v2.capture import shared as capture_shared\n        capture_shared(self,gate_up)",1)
capture_files['vllm/model_executor/models/qwen2_moe.py']=shared
manifest=[]
for rel,text in files.items():
 path=OUT/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);compile(text,str(path),'exec');manifest.append({'source':str(path),'target':'/opt/venv/lib/python3.12/site-packages/vllm/'+rel,'original_sha256':ORIGINAL_HASHES['vllm/'+rel],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
for rel,text in capture_files.items():
 path=OUT/'capture'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);compile(text,str(path),'exec');manifest.append({'source':str(path),'target':'/opt/venv/lib/python3.12/site-packages/'+rel,'original_sha256':ORIGINAL_HASHES[rel],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'capture_only':True})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Prepared',len(files),'diagnostic overlays')
