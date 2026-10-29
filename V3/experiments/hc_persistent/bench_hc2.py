"""Cold-L2 benchmark + parity: persistent HC mix vs current serving sequence (cuBLAS down + silu + V1 up_gate)."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/opt/v3/kernels')
sys.path.insert(0, '/opt/qwen38opt')
from hc_v3 import hc_mix_persistent
from kernels.hc_fused import hc_up_gate

torch.manual_seed(0)
dev = 'cuda'
H, S, R, N = 2560, 4, 320, 336
K = S * H
P = 168
COPIES = 24
wd = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(COPIES)]
wu = [torch.randn(K, R, device=dev, dtype=torch.bfloat16) * 0.05 for _ in range(COPIES)]
xs = [torch.randn(4, K, device=dev, dtype=torch.bfloat16) for _ in range(COPIES)]
counter = torch.zeros(1, device=dev, dtype=torch.int64)
down = torch.empty(4, N, device=dev, dtype=torch.bfloat16)
bott = torch.empty(4, R, device=dev, dtype=torch.bfloat16)
out = torch.empty(4, H, device=dev, dtype=torch.bfloat16)


def ref(x, wd_, wu_, T):
    d = F.linear(x, wd_)
    sc = (d[:, :R].float() / S).bfloat16().float()
    b = (sc * torch.sigmoid(sc)).bfloat16()
    o = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
    hc_up_gate[((T + 15) // 16, 160)](b, wu_, x, o, T, H, R, S, 16, num_warps=4)
    return d, b, o


def new(x, wd_, wu_, T, BK=512, warps=8):
    hc_mix_persistent[(P,)](x, wd_, wu_, down, bott, out, counter, T, K, N, R, S, H, P, BK, 16, num_warps=warps)
    return down[:T], bott[:T], out[:T]


def timeit(fn, iters=960):
    for i in range(COPIES * 2):
        fn(i % COPIES)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(COPIES):
            fn(i)
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay(); torch.cuda.synchronize()
    st.record()
    for _ in range(iters // COPIES):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / iters


for T in (1, 4):
    x = [v[:T].contiguous() for v in xs]
    d0, b0, o0 = ref(x[0], wd[0], wu[0], T)
    d1, b1, o1 = [t.clone() for t in new(x[0], wd[0], wu[0], T)]
    diff = lambda a, b: (a.float() - b.float()).abs().max().item()
    print(f'T={T} parity down={diff(d0, d1):.2e} bott={diff(b0, b1):.2e} (bott exact {torch.equal(b0, b1)}) '
          f'out={diff(o0, o1):.2e} out-mismatch-frac={(o0 != o1).float().mean().item():.4f}')
    print(f'  current serving sequence (cuBLAS down + torch silu + V1 up): {timeit(lambda i: ref(x[i], wd[i], wu[i], T)):.2f}us')
    for bk, wp in [(1024, 8), (2048, 8), (1024, 16), (2048, 16)]:
        print(f'  persistent BK={bk} warps={wp}: {timeit(lambda i: new(x[i], wd[i], wu[i], T, bk, wp)):.2f}us')
