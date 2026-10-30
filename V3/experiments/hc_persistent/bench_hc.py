"""Cold-L2 microbenchmark + parity for V3 HC kernels vs the V1 serving path.

Run inside the serving container: python /opt/v3/kernels/bench_hc.py
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/opt/v3/kernels')
sys.path.insert(0, '/opt/qwen38opt')
from hc_v3 import hc_down_silu, hc_up_gate_v3
from kernels.hc_fused import hc_up_gate

torch.manual_seed(0)
dev = 'cuda'
H, S, R, N = 2560, 4, 320, 336
K = S * H
COPIES = 24
wd = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(COPIES)]
wu = [torch.randn(K, R, device=dev, dtype=torch.bfloat16) * 0.05 for _ in range(COPIES)]
xs = [torch.randn(4, K, device=dev, dtype=torch.bfloat16) for _ in range(COPIES)]
part = torch.empty(32 * 16 * N, device=dev, dtype=torch.float32)
counter = torch.zeros(64, device=dev, dtype=torch.int32)


def ref_down_silu(x, w):
    d = F.linear(x, w)
    sc = (d[:, :R].float() / S).bfloat16().float()
    return d, (sc * torch.sigmoid(sc)).bfloat16()


def v3_down_silu(x, w, T, BN=16, BK=64, SPLITS=16, warps=4):
    down = torch.empty(T, N, device=dev, dtype=torch.bfloat16)
    bott = torch.empty(T, R, device=dev, dtype=torch.bfloat16)
    hc_down_silu[(triton_cdiv(N, BN), SPLITS)](x, w, part, counter, down, bott, T, K, N, R, S, BN, BK, SPLITS,
                                              num_warps=warps)
    return down, bott


def triton_cdiv(a, b):
    return (a + b - 1) // b


def v1_up(b, w, x, T):
    out = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
    hc_up_gate[((T + 15) // 16, 160)](b, w, x, out, T, H, R, S, 16, num_warps=4)
    return out


def v3_up(b, w, x, T, BH=8, BR=64, warps=4):
    out = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
    hc_up_gate_v3[(H // BH,)](b, w, x, out, T, H, R, S, BH, BR, num_warps=warps)
    return out


def timeit(fn, iters=480):
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


for T in (1, 2, 4):
    x = [v[:T].contiguous() for v in xs]
    d_ref, b_ref = ref_down_silu(x[0], wd[0])
    d_new, b_new = v3_down_silu(x[0], wd[0], T)
    up_ref = v1_up(b_ref, wu[0], x[0], T)
    up_new = v3_up(b_ref, wu[0], x[0], T)
    print(f'T={T} parity: down maxdiff={(d_ref.float() - d_new.float()).abs().max():.3e} '
          f'bott maxdiff={(b_ref.float() - b_new.float()).abs().max():.3e} '
          f'bott exact={torch.equal(b_ref, b_new)} up maxdiff={(up_ref.float() - up_new.float()).abs().max():.3e} '
          f'up exact={torch.equal(up_ref, up_new)}')
    t_ref_down = timeit(lambda i: ref_down_silu(x[i], wd[i]))
    t_lin = timeit(lambda i: F.linear(x[i], wd[i]))
    t_v1_up = timeit(lambda i: v1_up(b_ref, wu[i], x[i], T))
    print(f'  cuBLAS down GEMM alone {t_lin:.2f}us | ref down+silu (cuBLAS+torch) {t_ref_down:.2f}us | V1 up_gate {t_v1_up:.2f}us')
    for cfg in [(16, 64, 16, 4), (16, 128, 16, 4), (16, 64, 20, 4), (32, 64, 16, 4), (16, 64, 32, 4), (16, 128, 8, 8)]:
        BN, BK, SP, wp = cfg
        if (K // SP) % BK:
            continue
        t = timeit(lambda i: v3_down_silu(x[i], wd[i], T, BN, BK, SP, wp))
        print(f'  v3 down_silu BN={BN} BK={BK} splits={SP} warps={wp}: {t:.2f}us')
    for cfg in [(8, 64, 4), (8, 32, 4), (4, 64, 4), (16, 64, 8), (8, 64, 8), (4, 32, 2)]:
        BH, BR, wp = cfg
        t = timeit(lambda i: v3_up(b_ref, wu[i], x[i], T, BH, BR, wp))
        print(f'  v3 up_gate BH={BH} BR={BR} warps={wp}: {t:.2f}us')
