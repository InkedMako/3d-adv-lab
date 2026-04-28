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
- `libgl1`（OpenGL 支持，cv2 和 open3d 渲染依赖）
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

## 7. Windows 行尾与容器挂载说明

Windows 工作区下的 `.sh` 文件默认为 CRLF 行尾，直接在容器中运行会导致 `set: pipefail: invalid option name` 等错误。  
建议的做法：

```bash
# 方法1：运行时直接规范化行尾
docker run --rm -v "${PWD}:/workspace" -w /workspace 3d-adv-lab:cu124 bash -lc \
  "tr -d '\\r' < scripts/install_ubuntu_env.sh > /tmp/install.sh && bash /tmp/install.sh"

# 方法2：下载仓库后先转换所有 Shell 脚本为 LF
dos2unix scripts/*.sh
```

## 8. 无头容器（Headless）与渲染说明

在无桌面环境（如服务器或容器）中运行脚本时，渲染操作会因缺少显示服务而失败。  
当前 `minimal_pointcloud_smoke_test.py` 默认**不启用渲染**，只输出 `.npy` 与 `.ply` 文件。  

```bash
# 标准用法（推荐在容器中执行，无渲染）
docker run --rm -v "${PWD}:/workspace" -w /workspace 3d-adv-lab:cu124 bash -lc \
  "source /workspace/.venv/bin/activate && python scripts/minimal_pointcloud_smoke_test.py"

# 若需启用渲染预览（需图形环境，通常在本地 Linux 或借助 X11 forwarding）
python scripts/minimal_pointcloud_smoke_test.py --render
```

验证环境是否就绪：检查 `outputs/smoke_test` 目录下是否存在 `report.json`（各样本的统计数据）。

## 9. 备注

当前工作区中已经保留了安装脚本 [install_ubuntu_env.sh](../scripts/install_ubuntu_env.sh)、Python 依赖定义 `requirements.txt` 和 `environment.yml`。  
Docker 镜像构建时会自动补齐所有系统库和 Python 包，无需手工操作。