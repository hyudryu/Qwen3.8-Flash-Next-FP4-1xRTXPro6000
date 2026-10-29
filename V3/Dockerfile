# V3 = V1 image + draft-vocabulary patches. Build V1 first: docker build -t qwen38-fp4-tp1:v1 V1
ARG V1_IMAGE=qwen38-fp4-tp1:v1
FROM ${V1_IMAGE}
USER root
COPY patches /opt/qwen38-v3/patches
COPY scripts/install.py /opt/qwen38-v3/scripts/install.py
COPY draft_vocab/top49152.json /opt/qwen38-v3/draft_vocab/top49152.json
RUN /opt/venv/bin/python /opt/qwen38-v3/scripts/install.py --site-packages /opt/venv/lib/python3.12/site-packages
LABEL org.opencontainers.image.title="Qwen3.8 TP1 V3: pruned + dynamic MTP draft vocabulary, MTP depth 4"
