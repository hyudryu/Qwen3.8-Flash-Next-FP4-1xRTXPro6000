# V3: pruned + dynamic MTP draft vocabulary, MTP depth 4

V3 builds on V1 (V1 patches unchanged) and targets **single-stream (C1) decode latency**. It does not change the target model, its BF16 LM head, or any target kernel. Every change is on the speculative-draft side, so the served distribution is unchanged; only how often drafts are accepted, and what each step costs, changes.

## Result (interleaved confirmation, TP1, 450 W, one RTX PRO 6000)

| Workload (C1) | V1 production config | V3 | Change |
|---|---:|---:|---:|
| Coding, `V1/benchmarks/coding.py`, 1536-token outputs, 2 × 10 rounds | 266.6 tok/s | **291.4 tok/s** | **+9.3%** (paired 95% CI +7.2% to +11.4%; 10/10 rounds faster) |
| MTP accepted drafts per step (coding) | 2.13 | 2.56 | |
| General: prose, math, French, Chinese, JSON, story (greedy) | 218.5 tok/s | 212.3 tok/s | −2.8% |
| General (temperature 0.7) | 217.7 tok/s | 212.6 tok/s | −2.3% |

Design: base, V3, base, V3, each a fresh container on the same GPU with the production arguments and environment; each block ran the general check then 10 coding rounds. Round *i* uses the same prompt in every run. Per-prompt general ratios (greedy): essay 0.91, math 1.10, French 0.93, Chinese 0.83, JSON 1.03, story 0.95. Raw numbers: [`results/confirmation.json`](results/confirmation.json).

**The trade-off is real:** coding gets +9%, while free-form prose, especially Chinese, gets a little slower. If prose/multilingual chat dominates your traffic, use depth 3 with the dynamic vocabulary (`dyn49k-d3` below, roughly neutral on both) or stay on V1.

## What changed

1. **Pruned draft LM head (static).** The MTP draft head was a full 248,320-token NVFP4 GEMM costing about 226 µs per draft token, about 6% of every decode step at depth 3. V3 adds a second online-NVFP4 b12x head containing only 49,152 tokens: special tokens plus the highest-frequency tokens from a weighted calibration mix of model generations (English code/prose and 17 natural languages, disjoint from the benchmark prompts) and tokenized real code/docs (Python, TS, JS, C/C++, Rust, Markdown). Its logits are scattered into a persistent full-vocabulary buffer whose other columns are −inf.
2. **Dynamic request-context vocabulary.** A static list cannot know topic words (`asyncio`, `loader`, 骑行, `fuite`), which mostly appear in the prompt. The MTP forward already sees every prompt and accepted token, so a Triton kernel there maintains a 1024-slot ring of non-static tokens (atomic slot claims, eviction resets the column to −inf). In `compute_logits` a gather-GEMV scores the ring against the **target's BF16 LM head rows** (the same checkpoint weights the draft head quantizes; about 5 MB per call) and writes them into the same buffer. Both kernels are registered custom ops, so they are opaque to `torch.compile` and CUDA-graph safe.
3. **MTP depth 4 with an exact CUDA-graph size.** Production captures graph sizes `[1, 2, 4, 8, …]`. Depth 3 verifies exactly 4 rows, but depth 4 needs 5 and was padded to the 8-row graph, which is why the earlier depth-4 screen lost to depth 3. V3 adds sizes 5/10/15/20. With a cheap draft head, the fourth draft token now pays for itself.

**Why this is lossless:** speculative rejection sampling is exact for any draft distribution. Columns outside the static and dynamic sets have zero draft probability, so those tokens are never proposed but are still produced by verification and recovery sampling. Temperature > 0 was exercised (general check, sampled mode). The deployment is not bitwise-deterministic even unmodified: two runs of the unmodified V1 server diverged after about 89–271 characters on the same greedy prompts, because `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0`. So output identity is not a usable check; acceptance and throughput are.

## Where the time went (V1, depth 3, C1)

From a low-overhead graph-level nsys trace and a node-level trace on the replica:

- Decode step about 11.2 ms for about 3.1 tokens. Target verify graph about 8.6 ms; BF16 target LM head (eager, outside the graph) 0.79 ms; draft graphs about 1.56 ms, of which three full-vocabulary draft heads were about 0.68 ms.
- Target-graph critical path (slice attribution, [`experiments/profiling/critpath.py`](experiments/profiling/critpath.py)): routed MoE 31.7%, hyper-connections 18.6%, GDN `in_proj` 11.5%, `out_proj` 6.9%, GDN core 6.2%, other projections 5.7%, PyTorch elementwise glue 4.9%, attention core 3.4%, idle gaps 2.8%.
- Per-forward weight traffic is about 8.4 GB (experts about 2.9 GB for about 25 unique experts per layer at 4 rows, GDN projections about 2.1 GB, BF16 HC projections about 1.3 GB). That puts a bandwidth floor of about 5.6 ms under the 8.6 ms graph. MoE and the GDN/HC GEMMs already stream at 1.0–1.5 TB/s with a cold L2.
- **Benchmark trap:** a single layer's weights (about 13 MB) stay resident in the 128 MB L2 when a microbenchmark repeats them, but serving streams about 8 GB per forward. Microbenchmarks here rotate 24 weight copies (about 320 MB). L2-hot timings explain why some V1 microbenchmark gains did not appear in serving.

## Explored and rejected

Full table: [`results/exploration.json`](results/exploration.json). Coding numbers are mean tok/s; general numbers are greedy mean tok/s. "Static" means the pruned head alone.

| Config | Coding | General | Note |
|---|---:|---:|---|
| V1 production (MTP3) | 265–268 | 222.9 | baseline screens |
| Static 49K, MTP3 | 277.8 | 202.9 | code-weighted list; multilingual acceptance collapses |
| Static 16K / 24K / 32K, MTP3 | 274 / 272 / 278 | — | 5-round screens |
| Static 49K, MTP5 (exact graph) | 290.1 / 294.6 | 196.4 | best coding; prose regresses |
| Full draft head, MTP5 (exact graph) | 273.0 | 203.9 | depth alone does not pay; the pruned head is what makes depth work |
| Static 49K, MTP6 / MTP7 | 273.7 / 281.2 | — | acceptance plateaus (3.07 / 3.12) |
| Static + dynamic, MTP3 | 274.2 | 218.9 | roughly neutral on both |
| **Static + dynamic, MTP4 (V3)** | **291.8** | 215.5 | chosen |
| Static + dynamic, MTP5 | 290.9 | 205.1 | |
| Adaptive depth (`adaptive_speculative_tokens_window=8`, max 5), 49K / 98K | 272.5 / 267.8 | 209.7 / 208.8 | the controller settles near 4, and variable shapes cost more than they save |
| Persistent fused HC kernel + static 49K, MTP3 | 276.4 | — | see below |

**Persistent fused hyper-connection kernel** ([`experiments/hc_persistent`](experiments/hc_persistent)) is rejected. It is one resident-grid Triton launch for the small-row TP1 HC mix: down/inject GEMV with deterministic full-K rows and scaled SiLU, a generation grid barrier, then up-projection, BF16 gates and ordered stream mean, with the up-weights prefetched at kernel entry. It is bit-exact against cuBLAS plus the b12x/V1 path at 1/2/4 rows. A cold-L2 microbenchmark measured 12.8–14.0 µs against about 15.5 µs for the in-graph serving sequence. In serving, 10 paired rounds gave a ratio of 0.996, so it is not used.

**Not attempted:** an FP8/MXFP8 target LM head (it changes the target distribution). Retuning b12x MoE/GEMM plans: the b12x tuner already flushes L2 before racing candidates, and the MoE grid of 128 CTAs reflects its task count, not a mistuned choice.

## Calibration and caveats

- The static list ([`draft_vocab/top49152.json`](draft_vocab/top49152.json)) is regenerable with [`draft_vocab/tools`](draft_vocab/tools). Only token-ID counts are shipped, not generated text. Weights: generated EN code/prose 0.25, generated multilingual 0.30, Python 0.12, TS 0.08, JS 0.05, Markdown 0.08, C/C++ 0.05, Rust 0.07. Coverage was evaluated on held-out coding outputs and on outputs for the six general-check prompts. Using those prompts to choose K is mild tuning: coverage at K=49K is 98.5% (coding) and 85.3% (general) for static only, and 99.2% and 88.2% with context tokens ([`simulate_coverage.py`](draft_vocab/tools/simulate_coverage.py)).
- Only C1 was measured. At higher concurrency, verify batches grow by depth + 1 per request, so re-check C4/C8 before relying on depth 4 there.
- Coding acceptance depends on the workload: the benchmark's long, repetitive implementations-with-tests favor deep drafting. The general check is short (6 prompts × 800 tokens per mode), so it is a sanity check, not a quality evaluation.
- Production TP1/450 W serving arguments and environment are otherwise unchanged. The two V3 variables are `QWEN38_DRAFT_VOCAB` (path to the ID list; unset means the original code path) and `QWEN38_DRAFT_DYNAMIC` (ring slots, a multiple of 16; 0 disables).

## Reproduce

```bash
docker build -t qwen38-fp4-tp1:v1 V1
docker build -t qwen38-fp4-tp1:v3 V3
python3 V3/scripts/serve.py --port 8000           # TP1 on GPU0; V3/scripts/serve-args.json + serve.env
python3 V3/benchmarks/bench.py --label v3 --rounds 10
python3 V3/benchmarks/bench_general.py --label v3
```

The patches are strict and hash-checked against the pinned image ([`patches/manifest.json`](patches/manifest.json)): `scripts/install.py --check | --restore`. Install, then restore, then reinstall is byte-exact. Kernel unit test (GPU, inside the image): `python V3/tests/dynamic_vocab_kernels.py`.
