# ART + MMDetection3D 安装说明

本文档记录当前课题所需的最小可用环境，基于 Ubuntu 22.04 + CUDA 验证通过。

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

推荐在 Ubuntu 或 Windows WSL2 环境中执行安装。

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
- 若要真正进行 CUDA 训练，需要宿主机 NVIDIA 驱动正常

## 7. Windows 行尾说明

Windows 工作区下的 `.sh` 文件默认为 CRLF 行尾，直接在 Linux/WSL2 中运行会导致 `set: pipefail: invalid option name` 等错误。  
建议的做法：

```bash
# 方法1：运行时直接规范化行尾
tr -d '\r' < scripts/install_ubuntu_env.sh > /tmp/install.sh && bash /tmp/install.sh

# 方法2：下载仓库后先转换所有 Shell 脚本为 LF
dos2unix scripts/*.sh
```

## 8. 无头模式与渲染说明

在无桌面环境（如服务器）中运行脚本时，渲染操作会因缺少显示服务而失败。  
当前 `minimal_pointcloud_smoke_test.py` 默认**不启用渲染**，只输出 `.npy` 与 `.ply` 文件。  

```bash
# 标准用法（无渲染）
source .venv/bin/activate && python scripts/minimal_pointcloud_smoke_test.py

# 若需启用渲染预览（需图形环境）
python scripts/minimal_pointcloud_smoke_test.py --render
```

验证环境是否就绪：检查 `outputs/smoke_test` 目录下是否存在 `report.json`（各样本的统计数据）。

## 9. 前后端启动

### 9.1 后端服务启动

```bash
# 在 Linux/WSL2 环境中启动
source .venv/bin/activate
python backend/app.py

# 或直接在Windows Conda环境中启动
conda activate mm3d
python backend/app.py
```

后端服务地址：`http://localhost:5000`

### 9.2 前端服务启动

```bash
cd frontend
npm install
npm run dev
```

前端服务地址：`http://localhost:5173`

## 10. 备注

当前工作区中已经保留了安装脚本 [install_ubuntu_env.sh](../scripts/install_ubuntu_env.sh) 和 Python 依赖定义 `requirements.txt`。