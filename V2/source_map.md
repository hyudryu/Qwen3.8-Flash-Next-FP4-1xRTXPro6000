# Executed Qwen path

Paths below are relative to `/opt/venv/lib/python3.12/site-packages` in the pinned runtime. Runtime component commits and overlay hashes are stored under `results/`.

| Stage | Source and call path |
|---|---|
| Decoder dependencies | `vllm/models/qwen4_exp/nvidia/model.py`: decoder forward → attention hyperconnection combine/mix → attention or GDN → MLP hyperconnection combine/mix → MoE; output combination is delayed until the next boundary. PLE addition can require earlier materialization. |
| Hyperconnection | `vllm/models/qwen4_exp/nvidia/hyperconnection.py`: combine/normalization → down/injection projection → scaled SiLU → up projection → gated stream mean. The backend already fuses combine and normalization. V1 specializes up projection/gating for small rows. |
| Router and shared expert | `vllm/model_executor/models/qwen3_next.py`: `Qwen3NextSparseMoeBlock`; BF16 router, 512 experts, top-10, hidden width 2560, expert intermediate width 640. |
| Shared FFN | `vllm/model_executor/models/qwen2_moe.py`: `Qwen2MoeMLP.forward` → gate/up projection → `SiluAndMul` → down projection → sigmoid shared gate/product. The tested target shared down projection is MXFP8/A16; the draft shared expert is ineligible. |
| Routing/dispatch and overlap | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`: shared-expert scheduling → authoritative router/top-k → routed expert preparation/compute → stream join and shared/routed combination. |
| Expert preparation/scales | `flashinfer/experimental/b12x/_lib/quant/nvfp4_csf.py`, `.../csf_blocks_tuning.py`, `.../moe/fused_moe/_preparation.py`: prepared NVFP4 scales, route work, and execution state. V2 does not alter these. |
| Routed expert MMA | `flashinfer/experimental/b12x/moe/_shared/kernels/silu.py`, materialized NVFP4 phase kernels and prepared fused MoE dispatch. Production tracing identifies `MoEDynamicKernelSilu` as a major cost. |
| Shared projection preparation | `flashinfer/experimental/b12x/gemm/blockscaled/_linear.py`, `_preparation.py`, and `_a16.py`: packed MXFP8 rows/scales, exact prepared regimes, A16 GEMM/split-K. V2 reads the existing packed weights and requires the prepared down-projection mode to remain A16. |
| PLE | `vllm/models/qwen4_exp/nvidia/b12x_ple.py` and `ple_shared_table.py`: mapped host table access; separate from GPU-resident expert weights. No disk expert staging or SSD relocation is introduced. |
| Target execution | `vllm/v1/worker/gpu/model_runner.py` and `cudagraph_utils.py`: actual/padded input batch construction, graph dispatch, logits, sampling, and state updates. Host dispatch row counts are recorded outside graph replay. |
| MTP | `vllm/models/qwen4_exp/nvidia/mtp.py` and `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`: draft prefill, output-index compaction, and fused draft steps. Draft model input rows and compacted FFN rows differ. |
| Verification/rollback | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` and `input_batch.py`: rejection sampling, committed/rejected counts, and recurrent/KV state advancement. |

Recorded target verification dispatch includes 4/4 rows at C1 and 32/32 at C8. Draft first-pass model input can also have 32 rows, but its compacted FFN has 8 request rows; subsequent draft steps use 8. Transitional batches and padding are retained in the raw host histograms. Concurrency is not used as a proxy for executed rows.

The fusion runs only for uniform FULL decode graphs with compatible prepared modes and supported small row counts. C8 target verification at 32 rows follows the original path. A faster small-row shared projection cannot by itself establish a faster decoder: routed expert compute, dense projections, hyperconnections, attention/GDN, sampling, and draft work remain.
