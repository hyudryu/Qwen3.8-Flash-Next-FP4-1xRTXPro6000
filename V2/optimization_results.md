# V2 decision: do not promote

The split-K shared-FFN candidate did not improve C1 serving throughput over V1. Keep both candidates opt-in and restore the preserved V1 deployment. The 10% research objective was not achieved. V1 files remain unchanged.

## Main matched comparison

One RTX PRO 6000 Blackwell 96 GB, TP1, fixed 600 W cap, pinned model/runtime, MTP3. Four held-out task templates (coding, prose, tools), template-formatted 2,048-token inputs and natural EOS with a 1,024-token output cap. Of 720 main-comparison requests, 716 reached the cap and four ended naturally (966 tokens in V1; 921, 979 and 941 in V2). All cases are retained, with actual committed-token counts. Twenty paired batches per cell, ABBA source order, warmups excluded. GPU performance measurements use unprofiled processes without activation/logit copies. Values below are geometric means over paired batches; confidence intervals bootstrap paired log ratios (10,000 resamples).

| Cell | V1 committed tok/s | V2 committed tok/s | V2 vs V1 (95% CI) |
|---|---:|---:|---:|
| C1 cold request | 254.92 | 253.25 | -0.66% [-3.01, +1.87] |
| C1 warm request | 270.51 | 266.41 | -1.52% [-3.11, +0.15] |
| C8 cold request | 771.49 | 771.16 | -0.04% [-0.85, +0.82] |
| C8 warm request | 848.09 | 838.83 | -1.09% [-1.86, -0.30] |

Cold request throughput includes prefill. Warm request throughput includes the cached-prefix request and its decode tail; it is not the steady decode metric below. Requests use persisted token IDs, with tokenizer preparation outside the timed window.

| Cell | V1 steady decode tok/s | V2 steady decode tok/s | V2 vs V1 (95% CI) |
|---|---:|---:|---:|
| C1, cold prefix | 270.03 | 268.40 | -0.60% [-3.10, +2.12] |
| C1, warm prefix | 272.29 | 268.13 | -1.53% [-3.19, +0.22] |
| C8, cold prefix | 927.95 | 928.51 | +0.06% [-0.94, +1.06] |
| C8, warm prefix | 921.98 | 915.32 | -0.72% [-1.34, -0.06] |

Steady decode uses the common overlapping emission window after every stream has emitted and before the first stream finishes. It counts changes in cumulative committed-token usage. Chunks are not treated as tokens. API generation counters matched committed completion counts in all 160 measured batches across V1 and V2.

## Original and target-only controls

Original controls are **three descriptive matched batches per cell**, not a 20-pair confirmation. This table uses the same three repeat indices in each variant; it does not compare the original three-prompt mixture against the full 20-batch mixture above. All four original integration-file hashes match the unmodified pinned image. V1's four runtime overlay hashes match the preserved V1 package.

| Matched three-batch cell | Original request tok/s | V1 request tok/s | V2 request tok/s |
|---|---:|---:|---:|
| C1 cold | 241.23 | 251.61 | 248.30 |
| C1 warm | 260.55 | 258.45 | 247.24 |
| C8 cold | 684.24 | 772.01 | 773.62 |
| C8 warm | 741.14 | 855.95 | 839.86 |

The observed original-to-V2 difference includes V1's existing patches; it must not be attributed to the new V2 kernel. These workload-specific controls do not revise historical V1 results.

Target-only execution disables MTP only for a separate **original-source diagnostic**, with three batches per cell. Its geometric steady decode rates are 125.74/125.62 tok/s at C1 (cold/warm prefix) and 583.05/582.08 tok/s aggregate at C8. Its KV allocation differs because the draft model is absent. It is not a main MTP3 optimization result or a speculative-depth search.

## Actual implementation and trace changes

- The single-kernel activation/projection/gating prototype reduced a hot, repeated-weight local boundary by about 8%, but its instrumented production kernel took about 33.9 microseconds. It was rejected before promotion.
- The split-K variant uses a producer projection/prologue and a reduction/gating epilogue. In the verified C1 graph, they take approximately 4.99 and 1.42 microseconds under profiling. The original four-kernel boundary sums to approximately 9.43 microseconds in the reference trace. These instrumented component timings are not final serving throughput.
- C1 target graph kernel nodes change from 2,103 to 2,007: activation, original down projection, sigmoid and product are replaced by two kernels in each of 48 target layers. CPU graph-launch count remains one per target replay. The single-kernel prototype had 1,959 nodes.
- C8 target verification at 32 rows retains all 2,344 reference kernel nodes. The draft shared expert is ineligible and unchanged. Supported small decode rows are prepared before capture; prefill, mixed batches, unsupported modes and larger rows use reference forward.
- Activation and BF16 projection/gate intermediates disappear, but the split variant adds FP32 partial-sum traffic. One new 655,360-byte scratch buffer is reused across eligible layers; the original down projection did not use these FP32 partial sums. No expanded-weight/scale cache is reserved. Startup KV capacity remains 792,224 reported tokens for V1 and V2.
- The split projection uses 92 registers/thread and 24 KiB shared memory, 128 threads/CTA. Driver occupancy inspection reports a resource limit of four CTAs/SM (33.3% theoretical thread occupancy), not achieved occupancy. No persistent cross-CTA spin synchronization is introduced.

## Latency, acceptance and memory

The [serving summary](results/final-serving-summary.json) retains TTFT, per-stream decode ms/token, request P50/P95 latency, accepted drafts and committed tokens per request verification round. The [phase report](results/phase-profile.json) separates target verification, first draft pass and fused remaining draft steps using NVTX host descriptors, with actual and padded rows. Those phase times include profiler overhead; host graph API durations can include blocking and are not pure launch overhead.

The input/output graph catalog, checkpoint, BF16 rounding edges, mixed quantization and MTP3 remain fixed. Added persistent scratch is 640 KiB; no KV-capacity loss versus V1 was observed at the allocator granularity. All expert weights remain GPU resident; the 26.82 GiB PLE table remains mapped host memory. Prediction recall, CPU expert prefetch stalls and wasted expert-prefetch bytes are not applicable: V2 implements neither prediction nor expert offload.

Busy GPU0 telemetry (utilization at least 80%, includes warmups and both concurrency cells):

| Block | Median W | Median SM MHz | Median °C | Peak used MiB |
|---|---:|---:|---:|---:|
| v1-a | 454.3 | 2827 | 61 | 93598 |
| v1-a2 | 454.9 | 2827 | 62 | 93598 |
| v2-b1 | 450.3 | 2827 | 60 | 93594 |
| v2-b2 | 460.0 | 2827 | 63 | 93594 |

No power, clock, security, BIOS or IOMMU setting was changed. Sampling power is not a claim that the GPU draws its full 600 W limit.

## Validation status

Real recorded activations and checkpoint weights cover all 48 affected target layers and rows 1, 2, 4, 8, 16, 32, and 64: 336 replay cases per candidate. The SwiGLU intermediate is exact, including the BF16 rounding after SiLU and multiplication. Split projection/output maximum absolute difference is 0.0009765625; projection reduction order differs. This is not a full-model correctness certificate.

No new fusion-kernel decode JIT warnings were recorded. The pinned runtime emitted first-use auxiliary-state/update/grammar warnings during excluded warmup or diagnostic requests; these are retained in [the JIT check](results/decode-jit-check.json).

Standalone graph replay and shared-scratch changing-shape tests passed Compute Sanitizer memcheck with zero errors. The verified optimized model passed greedy and non-greedy coding smoke tests, top-p, structured JSON, tool-call parsing, and 16K/64K retrieval smoke checks.

Four forced-continuation cases cover 1,024 positions, with full-vocabulary raw logits captured internally through the optimized target path (no logprob request). Raw BF16 logits are preserved exactly in FP32 diagnostic storage, and comparisons use FP64 log-softmax. Prefix caches are reset before each case. The results do **not** establish numerical parity:

| Comparison | Top-1 agreement | Mean KL(reference‖candidate) | Mean KL(candidate‖reference) | Max logit error |
|---|---:|---:|---:|---:|
| Original vs original | 91.02% | 0.240001 | 0.232860 | 17.062500 |
| Original vs V1 | 91.02% | 0.319529 | 0.264301 | 18.843750 |
| V1 vs V2 | 94.14% | 0.158215 | 0.152694 | 16.281250 |
| Original vs V2 | 91.70% | 0.260582 | 0.215524 | 17.093750 |
| Target-only original repeat | 92.48% | 0.193454 | 0.193840 | 14.875000 |

Large differences remain even in unmodified-original repeats and without MTP. The diagnostic/runtime state behavior needs investigation; these values are neither an accepted correctness threshold nor evidence that V2 improves accuracy. Per-position and per-case outliers are retained in the JSON. Recurrent/KV and MTP rejection rollback parity are not certified by this experiment. Numerical promotion remains blocked regardless of the functional smoke passes. The teacher-forcing workload is a separate raw-prefix diagnostic, not the chat-template natural-generation throughput workload.

## Why the local fusion did not help the join

The shared branch runs on an auxiliary CUDA stream. In the reference C1 trace, all **4,800/4,800** aligned shared-output producers finished before their routed-output producer, with a median 51.46 microsecond readiness margin. The split trace shows the same for **4,656/4,656** layer instances, with a 58.32 microsecond margin. Shortening the already-ready branch does not advance the MoE output join. These are producer-end comparisons within each replay, not a sum of overlapping durations or a full event-edge critical-path proof.

The routed MoE kernel averages about 55.8 microseconds per layer at C1 and 211.5 at C8 in the modified instrumented trace. That branch, its route/scale preparation, and its relationship to dense/hyperconnection work are higher-value next targets. A persistent routed NVFP4 replacement and new HC/input-normalization fusion were not implemented here. The delivered scope is the shared small-row projection/prologue and gating boundary.

## Limits and next bottleneck

Nsight Compute hardware counters are blocked by `ERR_NVGPUCTRPERM`. Host policy was left unchanged. Achieved occupancy, Tensor Core utilization and measured memory bandwidth are unavailable; logical buffer accounting is not presented as measured traffic. Graph envelopes and summed kernel categories are reported separately and must not be added across overlapping work.

Routed NVFP4 expert execution remains a much larger measured GPU component than the modified shared boundary. Dense/hyperconnection work, attention/GDN, PLE and MTP remain. The local launch reduction did not establish an end-to-end gain. C4/C16 and 16K/64K performance confirmation/extended ablations are not run for this non-winning candidate; long-context smoke checks are correctness checks, not throughput confirmation.

Fusion implementation snapshot: `f3dfd88` on `experiment/v2-fused-ffn`. Runtime component commits, checkpoint revision and overlay hashes are in [provenance](results/provenance.json). See [serving summary](results/final-serving-summary.json), [raw timing samples](results/raw-serving.jsonl.gz), [raw metrics](results/raw-metrics.jsonl.gz), [phase envelopes](results/phase-profile.json), [reference join readiness](results/v1-join-readiness.json), and [split join readiness](results/split-join-readiness.json). Compressed raw CUDA kernel intervals and GPU telemetry are included under `results/`. Full native profiler databases remain local; the scripts reproduce collection. No weights, activation dumps, full logit tensors, private environment, or control tokens are published.

## Deployment and rollback

V2 was not promoted. GPU0's preserved V1 container is restored via SparkDeck, with the original command, environment, image, mounts and restart policy verified unchanged. GPU1 remains stopped. A request through `http://localhost:7878/v1` returned `READY` from `qwen3.8-flash-next`. Both GPU power caps remain 600 W. See [restoration checks](results/restoration.json) and [gateway verification](results/gateway-verification.json).

The image build and exact worker install/restore/reinstall roundtrip passed. The packaged image leaves fusion disabled by default. It has not been benchmarked as a separate full-model deployment. The two implemented shared-FFN candidates are isolated research paths; no new persistent routed NVFP4 kernel, HC normalization fusion, or full HC/input/output factorial ablation is claimed.
