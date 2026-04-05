FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]
