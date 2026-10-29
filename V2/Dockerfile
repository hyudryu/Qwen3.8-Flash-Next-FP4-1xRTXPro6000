ARG BASE_IMAGE=ghcr.io/local-inference-lab/vllm@sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29
FROM ${BASE_IMAGE}
USER root
COPY V1/kernels /opt/qwen38-v1/kernels
COPY V1/patches /opt/qwen38-v1/patches
COPY V1/scripts/install.py /opt/qwen38-v1/scripts/install.py
RUN /opt/venv/bin/python /opt/qwen38-v1/scripts/install.py --site-packages /opt/venv/lib/python3.12/site-packages
COPY V2/qwen_v2 /opt/qwen38-v2/qwen_v2
COPY V2/scripts/install.py /opt/qwen38-v2/scripts/install.py
RUN /opt/venv/bin/python /opt/qwen38-v2/scripts/install.py
ENV PYTHONPATH=/opt/qwen38-v2:/opt/qwen38-v1
ENV QWEN_V2_MODE=v2 QWEN_V2_SHARED_FUSION=0 QWEN_V2_CAPTURE=0 QWEN_V2_PROFILE=0
LABEL org.opencontainers.image.title="Qwen3.8 TP1 V2 source experiment (disabled by default)"
