"""Idle-worker hashes of loaded modules and callable/JIT source objects."""
import hashlib,inspect,json,marshal,pathlib,sys
from qwen_v2.diagnostics import OUTPUT

def run(label):
    suffixes=['vllm/models/qwen4_exp/nvidia/model.py','vllm/models/qwen4_exp/nvidia/hyperconnection.py','flashinfer/experimental/b12x/_lib/quant/nvfp4_csf.py','flashinfer/experimental/b12x/_lib/quant/csf_blocks_tuning.py','flashinfer/experimental/b12x/moe/fused_moe/_preparation.py','vllm/model_executor/models/qwen2_moe.py','qwen_v2/hooks.py','qwen_v2/fused_shared.py']
    records=[]
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file or not any(file.endswith(suffix) for suffix in suffixes):continue
        functions={}
        for key,value in vars(module).items():
            if inspect.isfunction(value) and value.__module__==name:
                functions[key]=hashlib.sha256(marshal.dumps(value.__code__)).hexdigest()
            elif inspect.isclass(value) and value.__module__==name:
                for method,fn in vars(value).items():
                    if inspect.isfunction(fn):functions[key+'.'+method]=hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()
                    elif isinstance(getattr(fn,'src',None),str):functions[key+'.'+method]=hashlib.sha256(fn.src.encode()).hexdigest()
            elif isinstance(getattr(value,'src',None),str):
                functions[key]=hashlib.sha256(value.src.encode()).hexdigest()
        records.append({'module':name,'file':file,'file_sha256':hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest(),'loaded_callable_hashes':functions})
    path=OUTPUT/(label+'.json');path.write_text(json.dumps({'scope':'File hashes plus loaded Python code/JIT source hashes; no weights or environment exported.','modules':records},indent=2));return {'path':str(path),'modules':len(records)}
