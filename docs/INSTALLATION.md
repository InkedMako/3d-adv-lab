# ART + MMDetection3D 安装说明

本文档记录当前课题所需的最小可用环境，基于 Ubuntu 22.04 + CUDA 容器验证通过。

## 1. 环境目标

目标是同时可用以下组件：

- ART 1.20.1
- MMCV 2.1.0
- MMEngine 0.10.7
- MMDetection 3.2.0
- MMDetection3D 1.4.0
- Open3D 0.19.0
- PyTorch 2.11.0+cu130

## 2. 推荐运行方式

由于当前工作区里的虚拟环境是 Linux 风格布局，推荐在 Ubuntu 或 Docker 容器中执行安装，而不是直接在 Windows PowerShell 中使用该 `.venv`。

推荐基础镜像：

- `nvidia/cuda:12.4.1-runtime-ubuntu22.04`

## 3. 系统依赖

在 Ubuntu 中先安装这些系统包：

- `python3`
- `python3-pip`
- `python3-venv`
- `python3-dev`
- `build-essential`
- `git`
- `ca-certificates`
- `curl`
- `cmake`
- `ninja-build`
- `libglib2.0-0`
- `libsm6`
- `libxext6`
- `libxrender1`
- `ffmpeg`

## 4. 安装步骤

在项目根目录执行：

```bash
bash scripts/install_ubuntu_env.sh
```

脚本会自动完成：

1. 安装 Ubuntu 系统依赖
2. 创建或复用 `/workspace/.venv`
3. 升级 `pip`、`setuptools`、`wheel`、`openmim`
4. 卸载冲突包 `mmcv-lite`、旧版 `mmcv`、旧版 `mmdet`
5. 安装兼容版本组合
6. 执行导入验证

## 5. 实际验证通过的版本组合

- `art==1.20.1`
- `mmcv==2.1.0`
- `mmengine==0.10.7`
- `mmdet==3.2.0`
- `mmdet3d==1.4.0`
- `open3d==0.19.0`
- `torch==2.11.0+cu130`

## 6. 关键说明

- `mmcv-lite 2.2.0` 与 `mmdet3d 1.4.0` 不建议混用，已替换为 `mmcv 2.1.0`
- `mmdet 3.3.0` 已降级为 `mmdet 3.2.0`
- 安装过程中容器可能提示未检测到 NVIDIA Driver，这不影响依赖安装，但会影响 GPU 训练
- 若要真正进行 CUDA 训练，需要宿主机 NVIDIA 驱动正常，且 Docker 运行时具备 GPU 直通能力

## 7. 最小测试

安装完成后，运行下面的导入检查：

```bash
python - <<'PY'
import importlib

mods = ["art", "mmcv", "mmengine", "mmdet", "mmdet3d", "open3d", "torch"]
for mod in mods:
    module = importlib.import_module(mod)
    version = getattr(module, "__version__", "unknown")
    print(f"{mod}: OK ({version})")
PY
```

## 8. 备注

当前工作区中已经保留了安装脚本 [install_ubuntu_env.sh](../scripts/install_ubuntu_env.sh)。如果后续你们要共享环境，建议再补一个 `requirements.txt` 或 `environment.yml`。