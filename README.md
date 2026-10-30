# Qwen3.8 Flash-Next FP4 on one RTX PRO 6000

**V1: source-level TP1 inference optimizations** for [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), tested on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB. The tested machine has 128 GB system RAM. This release contains actual CUDA/CuTe and Triton changes plus hash-checked inference integration patches. No TP2 results or two-GPU aggregate numbers are used here.

## V3 experiment

[V3 adds a pruned + dynamic MTP draft vocabulary and MTP depth 4](V3/README.md). It leaves the target model and its BF16 LM head untouched. In an interleaved TP1/450 W confirmation, C1 coding went from 266.6 to **291.4 tok/s (+9.3%, paired 95% CI +7.2% to +11.4%)**, while short general/multilingual prompts were 2–3% slower.

## V2 experiment

[V2 implements opt-in shared-FFN Triton fusion](V2/README.md), with production graph evidence and 20-pair C1/C8 benchmarks. It did **not** establish a C1 serving gain over V1 and remains disabled. See the [decision and raw results](V2/optimization_results.md). V1 below is preserved unchanged.

## Single-GPU coding throughput: 450W vs 600W

These are **V1 results**, with TP1 and MTP depth 3. Only one model instance was running. Median aggregate output tokens/s, three simultaneous-request batches per cell:

| Concurrency on one GPU | 450W | 600W | 600W change |
|---|---:|---:|---:|
| C1 | **272.90 tok/s** | **270.30 tok/s** | −0.95% |
| C4 | **673.70 tok/s** | **687.91 tok/s** | +2.11% |
| C8 | **975.95 tok/s** | **951.99 tok/s** | −2.46% |

**600W did not show a consistent throughput advantage.** Three batches and variable MTP acceptance are insufficient to establish a reliable effect this small. An earlier independent 600W screen measured C1 **273.66**, C4 **675.29**, and C8 **966.56 tok/s**; both screens are included in the raw results.

The workload asks for Python, TypeScript, Rust and Go implementations and tests. C1 uses the Python task; C4/C8 mix all four languages. Prompts contain 235–257 tokens, each output is capped at **1,536 tokens**, and thinking is disabled. Every measured request reached that cap. Temperature 0, seed 42, natural EOS, one warmup batch per concurrency, unique cache salts and verified zero cached prompt tokens. Outputs were generated for throughput testing; their coding tests were not executed.

Throughput is **total reported completion tokens / batch wall time**, including prompt processing and streaming overhead, measured directly against the engine. It is not GPU-only decode throughput. The incomplete gateway attempt was excluded after a request stalled while the GPU was idle. Backend counters verified exactly the requested number of completions and tokens on the single active engine.

At ≥80% GPU utilization, sampled draw was about 438W median with the 450W limit and 448W with the 600W limit; a 600W cap does not imply a 600W draw. These samples span the full campaign, including warmups. No clocks were locked. These short-context numbers are not a long-context coding or quality benchmark.

Data: [power comparison](V1/results/coding/power-comparison.json), [450W raw timing records](V1/results/coding/450w/), [600W confirmation](V1/results/coding/600w-confirm/), [telemetry](V1/results/coding/power-telemetry.json). Benchmarks were run on 2026-10-29. The other model deployment remained stopped throughout.

## What V1 changes

1. **Fused hyperconnection kernel:** combines the small TP1 up-projection, BF16 gate/product rounding and ordered FP32 stream mean. Used for at most four rows, hidden size 2560, four streams and rank 320. Larger or unsupported inputs use the original implementation.
2. **Faster lossless MoE scale expansion:** loads sparse exception metadata early, writes native scale storage directly, synchronizes the owning warp and patches exact bytes. The retained geometry uses 32 groups and no shared-memory exception staging.
3. **GPU route compaction:** builds a sorted unique list from the **actual router-selected expert IDs**, preserving invalid-ID handling and avoiding host counts/copies. It is enabled only for supported route counts; small decode and full-prefill paths retain their original handling.
4. **Prepared-execution integration:** installs the optimized decoder at the real b12x materialization point and retains/prepares the CUDA/Triton programs before compilation freezes, including CUDA Graph compatibility.

The original router decisions, NVFP4 expert MMA and weighted expert-output epilogue remain authoritative and unchanged. MoE expert weights are already GPU-resident. The **26.82 GiB CUDA-mapped host n-gram/PLE table is separate** from those weights. Expert prediction/prefetch prototypes did not yield an end-to-end benefit and are not enabled in V1. This is a source release, not a vLLM flag sweep; the tested launch arguments/environment are preserved for reproduction.

## What improved against the original source

Historical matched **single-GPU** tests, separate from the coding/power experiment above:

| Test | Original | V1 | Scope |
|---|---:|---:|---|
| C1, MTP3, 1K input | 222.48 tok/s | 226.53 tok/s | +1.82% median request throughput |
| C1, MTP3, 8K input | 219.63 tok/s | 218.60 tok/s | −0.47%; no universal gain |
| C1, no MTP, 1K input | 120.73 tok/s | 123.41 tok/s | +2.22%; auxiliary configuration |
| Prepared MoE, 8 rows | 33.47 µs | 30.75 µs | Fixed real routes; kernel microbenchmark |
| Prepared MoE, 16 rows | 66.58 µs | 63.89 µs | Fixed real routes; kernel microbenchmark |
| Prepared MoE, 32 rows | 164.96 µs | 162.38 µs | Fixed real routes; kernel microbenchmark |

MTP acceptance varied even between unmodified runs; the C1 MTP wall-latency confidence interval included zero. Some larger-concurrency historical cases regressed. Microbenchmark gains must not be interpreted as equivalent serving gains. The new power comparison measures V1 at two power limits; it does **not** compare original source against V1. [Historical full summary](V1/results/historical-single-gpu-summary.json), [implementation and limitations](V1/docs/engineering.md).

## Build and run V1

The base image is pinned to the exact tested b12x/vLLM fork. The source installer checks the original and patched SHA256 of all four integration files and refuses incompatible upstream versions.

```bash
git clone https://github.com/hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000.git
cd Qwen3.8-Flash-Next-FP4-1xRTXPro6000
docker build -t qwen38-fp4-tp1:v1 V1
python3 V1/scripts/serve.py --hf-cache "$HOME/.cache/huggingface"
```

Download the [weights](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) into that cache first. The tested environment uses `HF_HUB_OFFLINE=1`, revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, TP1, BF16 activations, mixed modelopt/NVFP4 weights, FP8 KV cache and MTP3. `serve.py` exposes port 8000 by default, selects only GPU0, and refuses Docker name/port conflicts rather than stopping another service. Inspect the exact launch using `--dry-run`; preserved arguments and non-secret environment are in [serve-args.json](V1/scripts/serve-args.json) and [serve.env](V1/scripts/serve.env). Model startup includes weight loading, kernel preparation and graph capture and can take several minutes. Access to the pinned GHCR base image and a compatible NVIDIA driver/container toolkit are required.

For SparkDeck, use the locally built `qwen38-fp4-tp1:v1` image for a **single-instance TP1** deployment, retain the provided arguments/environment, and select the cached checkpoint. The image contains the source patches; source bind mounts are unnecessary. This route was build/patch/smoke-tested; serving measurements used an existing instance with byte-identical source overlays. There was no second full-model deployment of the packaged image for these measurements.

## Reproduce the coding test

```bash
# Point both inference and metrics at the same single active instance.
python3 V1/benchmarks/coding.py \
  --base-url http://127.0.0.1:8000 --metrics-port 8000 \
  --concurrency 1 4 8 --rounds 3 --max-tokens 1536 \
  --output local-results/coding-450w
```

Use a distinct output directory for each run. Set the power limit explicitly before each run, verify it, allow the GPU to settle, and repeat with identical prompts. For example, with host administrator access:

```bash
sudo nvidia-smi -i 0 -pl 450
nvidia-smi -i 0 --query-gpu=power.limit,enforced.power.limit --format=csv
# Run the benchmark above, then repeat at 600W with a new output directory.
sudo nvidia-smi -i 0 -pl 600
```

The benchmark preserves launch settings, isolates prefix caches by request salt, validates server token counts and fails on competing request-counter changes. Power changes are runtime settings; no reboot-persistent power configuration is installed.

## Validation and rollback

```bash
docker run --rm -v "$PWD/V1/tests:/tests:ro" \
  --entrypoint /opt/venv/bin/python qwen38-fp4-tp1:v1 /tests/install_roundtrip.py
docker run --rm --gpus 'device=0' -v "$PWD/V1/tests:/tests:ro" \
  --entrypoint /opt/venv/bin/python qwen38-fp4-tp1:v1 /tests/gpu_smoke.py
```

Run GPU smoke checks on an idle GPU. The source install/restore/reinstall roundtrip is byte-exact for all four files; smoke checks cover HC parity and 12 route-compaction/CUDA Graph cases. Earlier real-model checks include 96 exact scale-plane cases across all 48 layers and 32/32 matching controlled token positions. HC fusion can change accumulation order: broader real-weight/synthetic-input maximum error was 0.00390625. Full-generation bitwise equivalence is not claimed.

For a Docker deployment, stop the V1 container and relaunch the preserved original command/environment using the original base image. For SparkDeck, stop V1, select the original pinned image and original settings, then start the original deployment. Checkpoint files are never modified. Container-local source rollback is also available through `/opt/qwen38-v1/scripts/install.py --restore`; restart the engine after restoring. Do not run modified and original full models simultaneously on the same GPU.

The repository's MIT license covers the original project material. Upstream licenses/attribution remain applicable to integration patches and dependencies; see [third-party notices](V1/docs/THIRD_PARTY_NOTICES.md).
