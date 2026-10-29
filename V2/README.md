# V2: Qwen shared-FFN source experiment

Experimental, opt-in, and **not promoted**. V1 is preserved unchanged. The target is `nvidia/Qwen3.8-Flash-Next-FP4-Local` on one RTX PRO 6000 Blackwell 96 GB, TP1. The main comparison retains MTP3, the checkpoint, quantization, sampling, and serving settings.

This implementation replaces a real small-row shared-FFN boundary, rather than changing launch settings. It is not a replacement for the routed NVFP4 expert matrix multiplication. See [measurements and decisions](optimization_results.md), [pinned provenance](results/provenance.json), and [source map](source_map.md).

The target shared expert uses MXFP8 weights and prepared BF16 activation GEMM at the tested decode sizes. The routed experts use NVFP4 and are GPU resident. The separate 26.82 GiB PLE table is mapped host memory.

Two independently disableable candidates are implemented in [fused_shared.py](qwen_v2/fused_shared.py):

- `single`: one Triton kernel consumes gate/up output, performs SwiGLU, decodes existing weight scales locally, projects, and applies the shared gate. It removes three launches per target layer. Its warm microbenchmark advantage did not transfer to production kernel timing.
- `split`: a split-K projection/prologue plus a reduction/gating epilogue. It uses a shared 640 KiB FP32 partial-sum workspace, without an expanded weight/scale cache. Successive layers depend on the prior MoE join, so the single worker can reuse that workspace. The new partial sums add memory traffic; eliminated activation/output intermediates do not imply lower total traffic.

Both retain BF16 rounding after SiLU, multiplication, projection, sigmoid, and the final product. Router decisions remain authoritative. Unsupported rows, prepared modes, prefill, mixed batches, and the differently quantized draft shared expert use the reference forward. Each kernel is compiled before CUDA Graph capture. No predictor, weight approximation, or expert dropping is used.

Build the opt-in image from the repository root:

```bash
docker build -f V2/Dockerfile -t qwen38-fp4-tp1:v2-experiment .
docker run --rm --entrypoint /opt/venv/bin/python -v "$PWD/V2/tests:/tests:ro" qwen38-fp4-tp1:v2-experiment /tests/install_roundtrip.py
```

The image installs the unchanged V1 patches and a hash-checked worker hook. Fusion defaults to disabled (`QWEN_V2_SHARED_FUSION=0`); `single` and `split` are explicit experiment selections. Measurements use preserved-runtime source overlays. The packaged image is build/integration tested, not claimed as a separately benchmarked full-model deployment.

To prepare a normal experiment from the preserved SparkDeck GPU0 container:

```bash
python3 V2/scripts/prepare_worker.py --container "$NATIVE_CONTAINER" --output V2/.runtime/overlay
python3 V2/scripts/run.py start --native-container "$NATIVE_CONTAINER" --deployment-id "$DEPLOYMENT_ID" --variant v2 --fusion split --overlay V2/.runtime/overlay --state V2/.runtime/experiment.json
```

The launcher requires the pinned image, TP1, MTP3, and GPU0. It preserves the native container, clones its serving settings, and refuses another running GPU0 container. It does not change power limits or host policy. Generated overlays retain the original source notices. Runtime state contains a diagnostic token when enabled and must stay private.

Rollback restores the preserved deployment through SparkDeck:

```bash
python3 V2/scripts/run.py rollback --state V2/.runtime/experiment.json
```

For authenticated idle-worker capture/replay and profiling, use `prepare_diagnostics.py` instead of `prepare_worker.py`, then pass its output to the same launcher. `--capture` adds bounded activation copies; `--profile` runs Nsight Systems with explicit profiler start/stop collection. Restart without those options for final performance measurements. Original-mode activation capture is intentionally rejected by this launcher to prevent accidentally reintroducing a V1 hyperconnection overlay.

Reproduce serving measurements with a persisted template-aware workload:

```bash
python3 V2/benchmarks/serving.py --variant v2 --workload V2/.runtime/coding-chat-workload.json --output V2/.runtime/measurements --concurrency 1 8 --repeats 20
```

Use the identical workload and repeat indices for each source variant. Cold cells have unique first-block contents and verify zero cached prompt tokens. Warm cells prime the exact prefix. Natural EOS is retained and capped completions are labeled. Completion counts come from cumulative committed-token usage, not streaming chunks. Requests send persisted chat-template token IDs; tokenizer preparation is outside the timed window. Cold request throughput includes prefill; steady decode uses a separately labeled overlapping committed-token window. The benchmark saves raw samples, metrics, output hashes, and GPU0 telemetry. Run variants in paired ABBA blocks; use `--repeat-start` for the second block.

`kernel_smoke.py` covers rows 1, 2, 4, 8, 16, 32, and 64 and repeated CUDA Graph replay. `functional.py` checks executable coding answers, greedy/non-greedy sampling, top-p, structured JSON, a tool call, and 16K/64K retrieval. `compare_logits.py` compares aligned full-vocabulary forced continuations in both KL directions. Replay instrumentation is a diagnostic and never part of throughput measurement.
