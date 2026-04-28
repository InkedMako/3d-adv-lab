# 小组实验统一环境配置手册

本文档用于给组内成员快速复现实验环境，目标是让所有人使用同一套依赖版本与执行方式，减少“能装但跑不通”的情况。

## 1. 适用范围

- 操作系统：Windows 10/11（推荐）
- 运行方式：Docker Desktop + WSL2
- 项目路径示例：E:/3d-adv-lab-clone
- 验证目标：能在容器内跑通最小测试并生成可视化结果

## 2. 统一版本基线

以下版本为本项目当前验证通过组合：

- Python 3.10（容器内 venv）
- torch 2.11.0+cu130
- torchvision 0.26.0+cu130
- torchaudio 2.11.0+cu130
- mmcv 2.1.0
- mmengine 0.10.7
- mmdet 3.2.0
- mmdet3d 1.4.0
- adversarial-robustness-toolbox 1.20.1
- open3d 0.19.0

依赖定义来源：

- environment.yml
- requirements.txt
- scripts/install_ubuntu_env.sh

## 3. 组员一次性准备

### 3.1 安装基础软件

- Docker Desktop（开启 WSL2 backend）
- Git

### 3.2 进入项目根目录（PowerShell）

~~~powershell
cd E:\3d-adv-lab-clone
~~~

## 4. 标准执行流程（必须按顺序）

### 4.1 构建镜像

~~~powershell
docker build -t 3d-adv-lab:cu124 .
~~~

### 4.2 安装 Python 环境与依赖（CRLF 安全方式）

Windows 下 shell 脚本可能是 CRLF 行尾，直接执行容易失败。统一使用下面命令：

~~~powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace 3d-adv-lab:cu124 bash -lc "tr -d '\r' < scripts/install_ubuntu_env.sh > /tmp/install_ubuntu_env.sh; chmod +x /tmp/install_ubuntu_env.sh; /tmp/install_ubuntu_env.sh"
~~~

### 4.3 运行最小可用测试（Smoke Test）

统一使用容器内解释器绝对路径，避免 PATH 差异导致 python 找不到：

~~~powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace 3d-adv-lab:cu124 bash -lc "/workspace/.venv/bin/python scripts/minimal_pointcloud_smoke_test.py --output-dir outputs/smoke_test"
~~~

### 4.4 生成可视化结果

~~~powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace 3d-adv-lab:cu124 bash -lc "/workspace/.venv/bin/python scripts/visualize_pointcloud_samples.py --input-root outputs/smoke_test --output-dir outputs/vis --dpi 150"
~~~

## 5. 成功判定标准

执行完成后应至少看到以下文件：

- outputs/smoke_test/report.json
- outputs/smoke_test/clean.npy
- outputs/smoke_test/adversarial.npy
- outputs/smoke_test/defended.npy
- outputs/vis/comparison_3d.png
- outputs/vis/comparison_2d_xy.png
- outputs/vis/summary.txt

若 summary 显示点云形状类似下列结果，可认为链路正常：

- clean: (2048, 3)
- adversarial: (1725, 3)
- defended: (1722, 3)

## 6. 常见问题与处理

### 6.1 执行命令“没有反应”

先检查 Docker 引擎是否可用：

~~~powershell
docker version
~~~

如果报错包含 open //./pipe/dockerDesktopLinuxEngine 或长时间卡住，按顺序执行：

~~~powershell
wsl --shutdown
docker context use desktop-linux
docker version
~~~

仍失败时重启 Docker Desktop 后重试。

### 6.2 bash 脚本报 pipefail 或语法异常

大概率是 CRLF 行尾问题，按 4.2 的 tr -d '\r' 方式执行，不要直接 bash scripts/install_ubuntu_env.sh。

### 6.3 容器提示未检测到 NVIDIA Driver

这不影响 CPU 跑通最小测试，但会影响 GPU 训练与推理。

## 7. 数据集准备（KITTI）

如果要继续做 KITTI 流程，先确保 data/kitti 目录结构完整，然后执行：

~~~powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace 3d-adv-lab:cu124 bash -lc "/workspace/.venv/bin/python scripts/prepare_kitti.py --data-root data/kitti"
~~~

成功后应有索引文件（示例）：

- data/kitti/kitti_infos_train.pkl
- data/kitti/kitti_infos_val.pkl
- data/kitti/kitti_dbinfos_train.pkl

## 8. 组内协作约定

- 一律以本手册命令为准，避免每人自行改版本
- 不提交 .venv、真实数据和大体积实验输出
- 结果统一落在 outputs 和 work_dirs
