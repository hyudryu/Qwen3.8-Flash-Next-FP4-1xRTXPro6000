# Qwen3.8 Flash-Next FP4 on one RTX PRO 6000

TP1 inference optimizations for [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (revision `b9ce18a6887e18c089fdea8509a33e63429a3300`) on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, 450 W), vLLM V2 model runner with b12x kernels and MTP speculative decoding.

## Latest changes (V3)

All changes are on the speculative-draft side. The target model, its BF16 LM head and the served output distribution are unchanged: speculative rejection sampling is exact for any draft distribution. Details, profiling and rejected experiments: [V3/README.md](V3/README.md).

1. **Pruned draft LM head.** The MTP draft head scored all 248,320 vocabulary tokens for every draft token. V3 scores a 49,152-token subset (special tokens plus the most frequent tokens in multilingual model generations and real code). The other columns are −inf.
2. **Dynamic request-context vocabulary.** Topic words that a fixed list misses (`asyncio`, 骑行) mostly come from the prompt. A Triton kernel tracks a 1,024-slot ring of prompt and output tokens outside the static list and scores them against the target's BF16 LM head rows. Both kernels are CUDA-graph safe custom ops.
3. **MTP depth 4 with exact CUDA-graph sizes.** Depth 4 verifies 5 rows per step, which used to be padded to the 8-row graph. V3 captures sizes 5/10/15/20, and the cheaper draft head makes the fourth draft token pay off.

## What improved against the original source

Paired on one GPU at 450 W: the unmodified pinned image and V3 were launched alternately (original, V3, original, V3) with the same serving arguments otherwise. V3 includes the [archived V1 kernel patches](V1/README.md) it is built on.

**Coding** (Python/TypeScript/Rust/Go implementation-and-test prompts, 235–257 tokens; 1,536 output tokens each; thinking disabled; temperature 0; unique cache salts; throughput = completion tokens / wall time including prompt processing):

| Coding, aggregate output tok/s | Original source | V3 | Change |
|---|---:|---:|---:|
| C1 | 257.4 | **282.4** | +9.7% |
| C4 | 669.5 | **659.0** | -1.6% |
| C8 | 943.4 | **979.0** | +3.8% |

_2 original + 1 V3 server launches, interleaved; 5 rounds per concurrency each; mean of all rounds._

**Prefill** (unique exact-length token-ID prompts, `max_tokens=1`, zero cached tokens verified, median of 3 per launch, [`V3/benchmarks/prefill.py`](V3/benchmarks/prefill.py)):

| Prefill | Original source | V3 | Change |
|---|---:|---:|---:|
| 1,024 tokens | 4,974 tok/s | 5,123 tok/s | +3.0% |
| 4,096 tokens | 14,549 tok/s | 14,663 tok/s | +0.8% |
| 16,384 tokens | 14,411 tok/s | 14,409 tok/s | -0.0% |
| 65,536 tokens | 14,009 tok/s | 14,020 tok/s | +0.1% |
| 131,072 tokens | 13,496 tok/s | 13,500 tok/s | +0.0% |
| C8 × 8,192 tokens (aggregate) | 14,476 tok/s | 14,497 tok/s | +0.1% |

The gain is concentrated at C1, where a fourth draft token is nearly free; at C4/C8 the larger verify batches use that spare capacity, and the C4/C8 differences are within run-to-run variation. Prefill is unchanged, since V3 only changes the draft step that runs after the prompt is processed. Short general/multilingual prompts at C1 were 2–3% slower than the V1 configuration (Chinese −17%, essay −9%, math +10%); see the [V3 README](V3/README.md) for that trade-off.

Raw data: [original vs V3](V3/results/vs-original/), [C1 confirmation against V1](V3/results/confirmation.json), [exploration](V3/results/exploration.json).

## Build and run

V3 is built on the V1 base image. Download the weights into your Hugging Face cache first.

```bash
git clone https://github.com/hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000.git
cd Qwen3.8-Flash-Next-FP4-1xRTXPro6000
docker build -t qwen38-fp4-tp1:v1 V1
docker build -t qwen38-fp4-tp1:v3 V3
python3 V3/scripts/serve.py --port 8000    # TP1 on GPU0; args in V3/scripts/serve-args.json, env in serve.env
```

The patches are hash-checked against the pinned image (`V3/scripts/install.py --check | --restore`); install, restore and reinstall is byte-exact. The kernel unit test runs inside the image: `python V3/tests/dynamic_vocab_kernels.py`.

## Archive

- [V1](V1/README.md): source-level b12x MoE and hyper-connection kernel patches (the base V3 builds on), with its 450 W/600 W results.
- [V2](V2/README.md): shared-FFN fusion experiment; no serving gain, disabled.

The repository's MIT license covers the original project material. Upstream licenses and attribution remain applicable to integration patches and dependencies; see [third-party notices](V1/docs/THIRD_PARTY_NOTICES.md).
