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
pip install -U pip setuptools wheel openmim

pip uninstall -y mmcv-lite || true
pip uninstall -y mmcv || true
pip uninstall -y mmdet || true

MMCV_WITH_OPS=1 pip install --no-build-isolation "mmcv==2.1.0"
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