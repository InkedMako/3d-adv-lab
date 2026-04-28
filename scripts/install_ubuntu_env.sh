#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  git \
  ca-certificates \
  curl \
  cmake \
  ninja-build \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  ffmpeg

if [ ! -x /workspace/.venv/bin/python ]; then
  python3 -m venv /workspace/.venv
fi

. /workspace/.venv/bin/activate

python -V
pip install -U pip wheel openmim
# Keep setuptools below 81 so pkg_resources remains available for legacy builds.
pip install "setuptools<81"

pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.11.0+cu130" \
  "torchvision==0.26.0+cu130" \
  "torchaudio==2.11.0+cu130"

# KITTI official metrics use numba CUDA kernels; pin a CUDA 12-compatible
# NVVM toolchain so the evaluation path works on a fresh environment.
pip install "nvidia-cuda-nvcc-cu12==12.4.131"

pip uninstall -y mmcv-lite || true
# Avoid pre-uninstalling mmcv/mmdet to prevent half-broken env on install failure.
pip install --no-build-isolation \
  -f https://download.openmmlab.com/mmcv/dist/cu130/torch2.11.0/index.html \
  "mmcv==2.1.0"
pip install \
  "mmengine==0.10.7" \
  "mmdet==3.2.0" \
  "mmdet3d==1.4.0" \
  "adversarial-robustness-toolbox==1.20.1" \
  "open3d==0.19.0" \
  "nuscenes-devkit" \
  "pycocotools" \
  "plyfile" \
  "shapely" \
  "trimesh" \
  "lyft-dataset-sdk"

python - <<'PY'
import importlib

mods = ["art", "mmcv", "mmengine", "mmdet", "mmdet3d", "open3d", "torch"]
for mod in mods:
    try:
        module = importlib.import_module(mod)
        version = getattr(module, "__version__", "unknown")
        print(f"{mod}: OK ({version})")
    except Exception as exc:
        print(f"{mod}: FAIL ({type(exc).__name__}: {exc})")
PY