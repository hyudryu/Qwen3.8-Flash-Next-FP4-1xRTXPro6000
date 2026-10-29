"""Unit test for the dynamic draft-vocab kernels, extracted from the installed (patched) mtp.py.

Run inside the V3 image on a GPU: python tests/dynamic_vocab_kernels.py [path/to/mtp.py]
"""
import sys, torch, triton, triton.language as tl
import importlib.util
_default = importlib.util.find_spec('vllm').submodule_search_locations[0] + '/models/qwen4_exp/nvidia/mtp.py'
src = open(sys.argv[1] if len(sys.argv) > 1 else _default).read()
start = src.index('@triton.jit\ndef _dyn_observe_kernel'); end = src.index('def _dyn_noop')
import importlib.util, tempfile, os
path = os.path.join(tempfile.mkdtemp(), 'dynk.py')
open(path, 'w').write('import torch, triton\nimport triton.language as tl\n' + src[start:end])
spec = importlib.util.spec_from_file_location('dynk', path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ns = vars(mod)
dev = 'cuda'; V = 1000; D = 32; R = 6; H = 256
slot_of = torch.full((V,), -1, dtype=torch.int32, device=dev); static = torch.arange(0, 500, device=dev); slot_of[static] = -3
ring = torch.full((D,), -1, dtype=torch.int32, device=dev); cursor = torch.zeros(1, dtype=torch.int64, device=dev)
full = torch.full((R, V), float('-inf'), dtype=torch.bfloat16, device=dev)
ids = torch.tensor([3, 600, 601, 600, 999, 5, 601, 700], device=dev)
ns['_dyn_observe_op'](ids, slot_of, ring, cursor, full); torch.cuda.synchronize()
r = sorted(t for t in ring.tolist() if t >= 0); print('ring after 1st:', r)
assert r == [600, 601, 700, 999], r
for t in r: assert ring[slot_of[t]].item() == t
# fill logits, then force eviction of all and check columns reset
w = torch.randn(V, H, dtype=torch.bfloat16, device=dev); h = torch.randn(R, H, dtype=torch.bfloat16, device=dev)
ns['_dyn_logits_op'](h, w, ring, full); torch.cuda.synchronize()
ref = (h.float() @ w[r].float().T)
got = full[:, r].float(); print('logit maxdiff', (ref - got).abs().max().item())
assert (ref - got).abs().max() < 0.5
assert torch.isinf(full[:, 650]).all() and torch.isinf(full[:, 10]).all()
ns['_dyn_observe_op'](torch.arange(501, 501 + D, device=dev), slot_of, ring, cursor, full); torch.cuda.synchronize()
assert all(torch.isinf(full[:, t]).all() for t in r if t not in range(501, 501 + D)), 'evicted columns not reset'
assert slot_of[999].item() == -1 and slot_of[700].item() == -1
assert sorted(ring.tolist()) == list(range(501, 501 + D)), sorted(ring.tolist())
print('dyn kernel tests passed')
