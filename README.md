# Qwen3.8 Flash-Next FP4 on one RTX PRO 6000

TP1 inference optimizations for [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (revision `b9ce18a6887e18c089fdea8509a33e63429a3300`) on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, 450 W), vLLM V2 model runner with b12x kernels and MTP speculative decoding.

## Latest: V3

[**V3**](V3/README.md) adds a pruned plus dynamic MTP draft vocabulary and MTP depth 4. The target model and its BF16 LM head are untouched, so the served distribution is unchanged.

### Coding throughput (TP1, 450 W)

Workload: Python/TypeScript/Rust/Go implementation-and-test prompts (235–257 tokens), 1,536 output tokens each, thinking disabled, temperature 0, unique cache salts. Throughput is total completion tokens divided by wall time, including prompt processing.

| Concurrency | V3 aggregate tok/s | Note |
|---|---:|---|
| **C1** | **291.4** | Paired against the V1 config in the same session: 266.6 → 291.4, **+9.3%** (95% CI +7.2% to +11.4%, 10/10 rounds faster) |
| C4 | 675.3 | Median of 3 batches; matches the V1 config's earlier 673.7 (not paired) |
| C8 | 948.9 | Median of 3 batches; the V1 config's earlier runs measured 952–976 (not paired) |

The gain is concentrated at low concurrency. A fourth draft token is nearly free when the GPU has idle capacity at C1, but at C4/C8 the larger verify batches use that capacity. Short general/multilingual prompts at C1 were 2–3% slower (Chinese −17%, essay −9%, math +10%); see the [V3 README](V3/README.md) for the trade-off and the depth-3 alternative.

### Prefill (TP1, 450 W)

Unique prompts sent as exact-length token IDs, `max_tokens=1`, zero cached tokens verified, median of 3 ([`V3/benchmarks/prefill.py`](V3/benchmarks/prefill.py)):

| Prompt tokens | Time to first token | Prefill tok/s |
|---:|---:|---:|
| 1,024 | 0.20 s | 5,051 |
| 4,096 | 0.27 s | 15,412 |
| 16,384 | 1.08 s | 15,145 |
| 65,536 | 4.47 s | 14,664 |
| 131,072 | 9.38 s | 13,974 |
| 8 × 8,192 concurrent | 4.40 s (last request) | 14,885 aggregate |

At 1K about 0.2 s per request is fixed overhead (on top of the prefill time). V3 does not change prefill; its draft-side changes run only after the prompt is processed.

Raw data: [C1 confirmation](V3/results/confirmation.json), [exploration](V3/results/exploration.json), [C4/C8 and prefill](V3/results/serving/).

## Build and run

V3 is built on the V1 base image, which contains the b12x MoE and hyper-connection kernel patches. Download the weights into your Hugging Face cache first.

```bash
git clone https://github.com/hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000.git
cd Qwen3.8-Flash-Next-FP4-1xRTXPro6000
docker build -t qwen38-fp4-tp1:v1 V1
docker build -t qwen38-fp4-tp1:v3 V3
python3 V3/scripts/serve.py --port 8000    # TP1 on GPU0; args in V3/scripts/serve-args.json, env in serve.env
```

The patches are hash-checked against the pinned image (`V3/scripts/install.py --check | --restore`); install, restore and reinstall is byte-exact. The kernel unit test runs inside the image: `python V3/tests/dynamic_vocab_kernels.py`.

The repository's MIT license covers the original project material. Upstream licenses and attribution remain applicable to integration patches and dependencies; see [third-party notices](V1/docs/THIRD_PARTY_NOTICES.md).
