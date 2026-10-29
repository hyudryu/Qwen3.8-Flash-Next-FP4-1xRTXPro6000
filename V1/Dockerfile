ARG BASE_IMAGE=ghcr.io/local-inference-lab/vllm@sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
FROM ${BASE_IMAGE}
USER root
COPY kernels /opt/qwen38-v1/kernels
COPY patches /opt/qwen38-v1/patches
COPY scripts/install.py /opt/qwen38-v1/scripts/install.py
RUN /opt/venv/bin/python /opt/qwen38-v1/scripts/install.py --site-packages /opt/venv/lib/python3.12/site-packages
ENV PYTHONPATH=/opt/qwen38-v1
LABEL org.opencontainers.image.title="Qwen3.8 TP1 source optimizations V1"
