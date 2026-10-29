"""V3 hyperconnection decode kernels (TP1, <=4 rows).

hc_down_silu: down/inject GEMV + deterministic split-K reduction + scaled SiLU in
one launch. Replaces cuBLAS split-K GEMM, splitKreduce and b12x ScaledSilu.
Numerics: FP32 accumulation, BF16 down output, then bf16(x/S), bf16(s*sigmoid(s)).

hc_up_gate_v3: V1 hc_up_gate math with a wider grid so cold-L2 weight streaming
keeps enough bytes in flight.
"""
import triton
import triton.language as tl


@triton.jit
def hc_down_silu(x, w, part, counter, down, bott, T,
                 K: tl.constexpr, N: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = pn * BN + tl.arange(0, BN)
    kchunk: tl.constexpr = K // SPLITS
    acc = tl.zeros((16, BN), tl.float32)
    for kb in tl.static_range(0, kchunk, BK):
        k = ps * kchunk + kb + tl.arange(0, BK)
        a = tl.load(x + rows[:, None] * K + k[None, :], mask=rows[:, None] < T, other=0.)
        b = tl.load(w + cols[None, :] * K + k[:, None], mask=cols[None, :] < N, other=0.)
        acc = tl.dot(a, b, acc)
    pmask = (rows[:, None] < T) & (cols[None, :] < N)
    tl.store(part + ps * 16 * N + rows[:, None] * N + cols[None, :], acc, mask=pmask)
    tl.debug_barrier()
    ticket = tl.atomic_add(counter + pn, 1, sem='acq_rel', scope='gpu')
    if ticket == SPLITS - 1:
        total = tl.zeros((16, BN), tl.float32)
        for s in tl.static_range(SPLITS):  # fixed order => deterministic
            total += tl.load(part + s * 16 * N + rows[:, None] * N + cols[None, :], mask=pmask, other=0.,
                             cache_modifier='.cg')
        d = total.to(tl.bfloat16)
        tl.store(down + rows[:, None] * N + cols[None, :], d, mask=pmask)
        sc = (d.to(tl.float32) / S).to(tl.bfloat16).to(tl.float32)
        act = (sc * tl.sigmoid(sc)).to(tl.bfloat16)
        tl.store(bott + rows[:, None] * R + cols[None, :], act, mask=pmask & (cols[None, :] < R))
        tl.atomic_xchg(counter + pn, 0)


@triton.jit
def hc_mix_persistent(x, wd, wu, down, bott, out, counter, T,
                      K: tl.constexpr, N: tl.constexpr, R: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                      P: tl.constexpr, BK: tl.constexpr, BH: tl.constexpr):
    """Whole TP1 HC mix (<=4 rows) in one resident-grid launch of P CTAs.

    Phase A: CTA p computes down rows 2p, 2p+1 over full K (deterministic, no split-K), then the scaled
    SiLU bottleneck. Its phase-B up-projection slice is loaded at entry so that the second weight
    stream overlaps phase A. Phase B: up projection, BF16 gates and ordered stream mean.
    `counter` is a monotonically increasing int64 (generation barrier; never reset).
    """
    pid = tl.program_id(0)
    # Issue this CTA's phase-B weight loads first so they stream during phase A.
    h = pid * BH + tl.arange(0, BH)
    hmask = h < H
    c = (tl.arange(0, S)[:, None] * H + h[None, :]).reshape((S * BH,))
    cmask = (tl.arange(0, S)[:, None] * 0 + hmask[None, :]).reshape((S * BH,)) > 0
    r0 = tl.arange(0, 256)
    r1 = 256 + tl.arange(0, 64)
    b0 = tl.load(wu + c[None, :] * R + r0[:, None], mask=cmask[None, :], other=0.)
    b1 = tl.load(wu + c[None, :] * R + r1[:, None], mask=cmask[None, :], other=0.)
    t16 = tl.arange(0, 16)
    v = tl.load(x + t16[:, None] * K + c[None, :], mask=(t16[:, None] < T) & cmask[None, :], other=0.)

    t4 = tl.arange(0, 4)
    rows = pid * 2 + tl.arange(0, 2)
    rmask = rows < N
    acc = tl.zeros((4, 2, BK), tl.float32)
    for kb in tl.static_range(0, K, BK):
        k = kb + tl.arange(0, BK)
        a = tl.load(x + t4[:, None] * K + k[None, :], mask=t4[:, None] < T, other=0.).to(tl.float32)
        w = tl.load(wd + rows[:, None] * K + k[None, :], mask=rmask[:, None], other=0.).to(tl.float32)
        acc += a[:, None, :] * w[None, :, :]
    d = tl.sum(acc, 2).to(tl.bfloat16)  # [4, 2]
    omask = (t4[:, None] < T) & rmask[None, :]
    tl.store(down + t4[:, None] * N + rows[None, :], d, mask=omask)
    sc = (d.to(tl.float32) / S).to(tl.bfloat16).to(tl.float32)
    act = (sc * tl.sigmoid(sc)).to(tl.bfloat16)
    tl.store(bott + t4[:, None] * R + rows[None, :], act, mask=omask & (rows[None, :] < R))

    # Generation grid barrier.
    tl.debug_barrier()
    ticket = tl.atomic_add(counter, 1, sem='release', scope='gpu')
    target = (ticket // P + 1) * P
    while tl.atomic_add(counter, 0, sem='acquire', scope='gpu') < target:
        pass
    tl.debug_barrier()

    a0 = tl.load(bott + t16[:, None] * R + r0[None, :], mask=t16[:, None] < T, other=0., cache_modifier='.cg')
    a1 = tl.load(bott + t16[:, None] * R + r1[None, :], mask=t16[:, None] < T, other=0., cache_modifier='.cg')
    up = tl.dot(a0, b0)
    up = tl.dot(a1, b1, up)
    logits = up.to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(logits).to(tl.bfloat16)
    product = (gate * v).to(tl.bfloat16).to(tl.float32).reshape((16, S, BH))
    total = tl.sum(tl.where(tl.arange(0, S)[None, :, None] == 0, product, 0.), 1)
    for stream in tl.static_range(1, S):
        total += tl.sum(tl.where(tl.arange(0, S)[None, :, None] == stream, product, 0.), 1)
    tl.store(out + t16[:, None] * H + h[None, :], total / S, mask=(t16[:, None] < T) & hmask[None, :])


@triton.jit
def hc_up_gate_v3(x, w, norm, out, T, H: tl.constexpr, R: tl.constexpr, S: tl.constexpr,
                  BH: tl.constexpr, BR: tl.constexpr):
    t = tl.arange(0, 16)
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    c = (tl.arange(0, S)[:, None] * H + h[None, :]).reshape((S * BH,))
    acc = tl.zeros((16, S * BH), tl.float32)
    for base in tl.static_range(0, R, BR):
        r = base + tl.arange(0, BR)
        a = tl.load(x + t[:, None] * R + r[None, :], mask=t[:, None] < T, other=0.)
        b = tl.load(w + c[None, :] * R + r[:, None])
        acc = tl.dot(a, b, acc)
    logits = acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(logits).to(tl.bfloat16)
    v = tl.load(norm + t[:, None] * (S * H) + c[None, :], mask=t[:, None] < T, other=0.)
    product = (gate * v).to(tl.bfloat16).to(tl.float32).reshape((16, S, BH))
    total = tl.sum(tl.where(tl.arange(0, S)[None, :, None] == 0, product, 0.), 1)
    for stream in tl.static_range(1, S):
        total += tl.sum(tl.where(tl.arange(0, S)[None, :, None] == stream, product, 0.), 1)
    tl.store(out + t[:, None] * H + h[None, :], total / S, mask=t[:, None] < T)
