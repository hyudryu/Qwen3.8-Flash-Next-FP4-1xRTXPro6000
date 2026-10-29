# V1 engineering notes

V1 ships the retained source path: three kernel modules, four strict integration patches, a reproducible container build, a single-GPU coding benchmark and rollback. Routing instrumentation and expert predictors were research experiments, not retained production dependencies.

## Execution path

The image implements this model internally as the NVIDIA `qwen4_exp` variant. The path is `Qwen3NextSparseMoeBlock` → authoritative fused top-k router / MoERunner → `Nvfp4CsfMoEMethod` → b12x prepared MoE execution. The 48 target MoE layers have 512 experts, top-10 routing, hidden size 2560 and intermediate size 640. Prepared expert matrices and their compressed scale payloads reside in GPU memory. Native scale-expansion scratch is a shared 100 MiB + 50 MiB pair across layers. The host PLE table is distinct from expert weights.

Patch targets, relative to the image's Python site-packages:

- `vllm/models/qwen4_exp/nvidia/hyperconnection.py`: dispatch the eligible small-row HC path to `kernels/hc_fused.py`.
- `flashinfer/experimental/b12x/_lib/quant/nvfp4_csf.py`: execute lossless native scale decoding with the retained prepared programs.
- `flashinfer/experimental/b12x/_lib/quant/csf_blocks_tuning.py`: materialize the selected custom scale path and prepare the route compactor.
- `flashinfer/experimental/b12x/moe/fused_moe/_preparation.py`: retain the custom programs in the actual execution plan before compilation freezes.

The final point matters: optimizing a declaration-time decoder alone does not change serving when b12x constructs a different runtime decoder during materialization. That error was discovered in development using Nsight Systems and corrected before V1. An actual serving trace verified custom `MetadataPrefetchPair` and `compact_experts` kernels at C8; the C1 scale path intentionally retains the original pair kernel.

## Kernel changes

HC fusion preserves intermediate BF16 rounding and ordered FP32 stream accumulation. It eliminates a launch and gate-logit traffic for supported ≤4-row calls. Live prepared-module tests measured roughly 4.37–4.49 µs originally versus 2.70–3.08 µs fused. The original implementation handles larger/unsupported inputs.

Scale decoding preloads exception bounds/first records, expands base/native scales, synchronizes each owning warp and patches exact bytes directly in global memory. The selected geometry uses 32 groups and no shared exception staging. Actual runtime layouts are query-dependent; V1 does not compare against an artificially slow fixed-layout baseline. Actual prepared reference versus modified scale-and-route microbenchmarks included about 7.87→5.52 µs at 8 rows, 8.72→6.81 µs at 16, 13.95→9.85 µs at 32 and 36.99→14.21 µs at 64. Small-row scale paths remain unchanged.

The Triton compactor validates IDs in their original integer width, constructs a histogram/prefix scan, emits sorted unique actual expert IDs and pads with −1. It prevents invalid large int64 IDs from wrapping into valid experts. Supported mid-sized route counts use this path; small/full-prefill paths retain their existing handling. No prediction replaces authoritative router choices.

A realistic prepared-layer benchmark includes the following expert operation and cache effects, not just isolated expansion. At 8/16/32 rows, complete MoE time changed 33.47→30.75 / 66.58→63.89 / 164.96→162.38 µs. Expert MMA and weighted reduction remain unchanged. Fixed-route differences were near the original atomic-repeat noise, not full bitwise arithmetic equality.

Nsight Compute on a runtime-layout 32-row scale case measured 20.61→13.70 µs, 622→929 GB/s DRAM bandwidth, 32 KiB→0 dynamic shared memory and 42.1%→89.9% achieved occupancy, with 40 registers unchanged. Profiling times differ from uninstrumented event timings. Original expert GEMM profiling was bandwidth-heavy (1.21 TB/s, 70.5% DRAM utilization) and was not replaced. These are specific observations, not universal speedup claims.

## Prediction result

Causal episode-held-out evaluation tested frequent-expert, previous-token, historical-transition, cross-layer and lightweight learned predictors. Earlier executed-layer features were available before the target router; temporal history was frozen across verification batches. On matched target layers 1–47, top-10 recall was approximately 9.16% frequent, 18.95% previous-token, 14.03% historical, 31.99% cross-layer, 39.12% selected-ID ridge and 60.78% full-logit ridge. Full-set hits remained low; the full-logit model cost roughly 6.4–6.7 µs. These were routing-proposal traces, not an accepted-token-only corpus.

All experts were GPU-resident. Predictive L2 hints slowed the tested pipeline even with oracle expert IDs. No CPU→GPU expert prefetch/cache was added. A private expanded-scale cache also cost roughly 6.9 GiB, reduced KV capacity and regressed a long-context target, so it was rejected. Top-k replacement prototypes were slower and were rejected. V1 retains neither predictor nor expanded private cache.

## Correctness and evidence

- Scale bytes: 96 actual runtime-plane cases across 48 target layers are exact; additional adversarial/materialization cases were tested during development.
- HC: live small-row module differences ≤0.00098; broader real-weight/generated-input tests up to 0.00390625. Synthetic input tests are not recorded real activations.
- Controlled short generation: 32/32 token positions matched the reference. Unmodified full generations and tail probabilities also varied; full-generation bitwise/distribution parity is not established by these checks.
- Representative arithmetic, JSON/tool-call, multilingual, long-retrieval and code-syntax screens passed. Code syntax checks are not functional coding evaluation.
- The packaged image builds successfully. All four original/patched hashes match and install→restore→reinstall is exact. Standalone GPU smoke tests cover two HC geometries and 12 exact compaction/graph cases.
- Serving benchmarks ran on the existing single-GPU instance with source bytes identical to this package. The built image was tested without loading another full model. This distinction avoids claiming an unperformed deployment test.

Selected numerical evidence is under `../results/`. Routing tensors, weight exports, complete upstream packages, credentials, private Docker inspections and binary profiler captures are not included.

## Power experiment limits

GPU0 was the only model instance running; GPU1's deployment stayed stopped. Power order was initial 600W screens, then 450W C1/C4/C8, then 600W confirmation. A warmup was excluded for each concurrency, and unique cache salts prevented repeated prompts benefiting from prefix caching. Before/after backend counters reject competing traffic and token mismatches. All prompts and per-request timings/usage/output hashes are published; generated output text is omitted.

The main table compares the final two matched three-batch blocks. Earlier 600W screens are also published, not discarded. Thermal settling was 10 seconds after each limit change; no clock locking, randomized power order or many-repeat statistical study was done. MTP acceptance and run order can affect these small differences. Power telemetry is sampled at about 1 Hz across the campaign, including warmups, and is not an exact per-token energy measurement. No causal power-performance improvement is claimed.
