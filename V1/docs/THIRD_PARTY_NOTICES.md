# Third-party notices

V1 modifies vLLM and the b12x integration supplied by the pinned `ghcr.io/local-inference-lab/vllm` image. It distributes integration diffs rather than copies of the full upstream modules. Original source headers, authorship and dependency licenses remain intact when patches are applied.

The vLLM hyperconnection source carries `SPDX-License-Identifier: Apache-2.0` and copyright attribution to the vLLM project contributors. An Apache 2.0 license copy is included as `Apache-2.0.txt`. See the pinned image for the accompanying FlashInfer/b12x, NVIDIA CUTLASS/CuTe, CUDA, PyTorch and Triton notices and license terms. These dependencies are not relicensed by this repository's MIT license.

The model weights are downloaded separately from `nvidia/Qwen3.8-Flash-Next-FP4-Local`; no weights, embeddings or extracted checkpoint tensors are redistributed here. Their own model/repository terms apply.
