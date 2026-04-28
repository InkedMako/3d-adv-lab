# 3D Adversarial Lab

本项目用于开展 ART + MMDetection3D 的对抗鲁棒性实验，当前以多模态 3D 感知（点云 + 图像）为主线。

## 1. 项目目录总览

3d-adv-lab/
- configs/
- data/
- docs/
- outputs/
- scripts/
- work_dirs/
- Dockerfile
- environment.yml
- requirements.txt
- README.md

## 2. 目录作用与文件格式要求

### 2.1 configs/

作用：实验配置文件，统一管理数据集、模型、攻击、防御和运行参数。

子目录：
- configs/base/：基础配置模板（供其他配置继承）
- configs/pointcloud/：点云任务配置
- configs/multimodal/：多模态任务配置

文件类型要求：
- 必须：.py（配置脚本）
- 可选：.md（目录说明）

命名建议：
- *_template.py（模板）
- *_baseline.py（可复现实验基线）
- *_expXX.py（批量实验）

### 2.2 data/

作用：数据集存放与处理中间产物。

子目录：
- data/raw/：原始数据（下载后不改动）
- data/processed/：预处理后的数据（采样、格式转换、索引文件）
- data/downloads/：下载缓存包（zip、tar 等）

文件类型要求：
- 原始数据：.bin, .pcd, .ply, .jpg, .png, .txt
- 标签与索引：.txt, .json, .pkl
- 压缩包：.zip, .tar, .gz

注意：
- data 目录默认不上传真实大数据到 GitHub，仅保留目录结构与说明文件。

### 2.3 docs/

作用：项目文档、方法说明、实验流程、安装和排错指南。

当前文件：
- docs/INSTALLATION.md
- docs/TEAM_ENV_SETUP.md
- docs/POINTCLOUD_TESTING_GUIDE.md
- docs/DATASET_ATTACK_DEFENSE_WORKFLOW.md
- docs/MULTIMODAL_DIRECTION.md

文件类型要求：
- 必须：.md
- 可选：.pdf, .docx（建议最终提交材料另存）

### 2.4 outputs/

作用：实验输出结果（样本、报告、可视化）。

子目录：
- outputs/pipeline_test/：统一流程脚本的测试输出
- outputs/smoke_test/：最小烟雾测试输出
- outputs/vis/：可视化图像与摘要

文件类型要求：
- 样本：.npy
- 报告：.json, .txt
- 可视化：.png, .jpg

当前示例文件：
- clean.npy, adversarial.npy, defended.npy
- report.json
- comparison_3d.png, comparison_2d_xy.png, summary.txt

注意：
- outputs 目录默认不上传真实实验大文件到 GitHub，仅保留目录结构与说明文件。

### 2.5 scripts/

作用：可执行脚本（安装、测试、流水线、可视化）。

当前脚本：
- scripts/install_ubuntu_env.sh
- scripts/minimal_pointcloud_smoke_test.py
- scripts/run_pointcloud_pipeline.py
- scripts/visualize_pointcloud_samples.py

文件类型要求：
- Python 脚本：.py
- Shell 脚本：.sh
- 说明文档：.md

脚本规范建议：
- 提供命令行参数
- 输出写入 outputs 或 work_dirs
- 错误信息清晰、可复现

### 2.6 work_dirs/

作用：训练过程和模型推理产生的工作目录。

常见内容：
- 日志：.log, .json
- 权重：.pth, .pt, .ckpt
- 临时中间文件

文件类型要求：
- 以训练框架产出为准（通常含 .pth/.log/.json）

注意：
- work_dirs 默认不上传大文件到 GitHub，仅保留目录结构与说明文件。

## 3. 根目录文件作用

- Dockerfile：容器化环境定义
- environment.yml：Conda 环境定义
- requirements.txt：pip 依赖列表
- .gitignore：忽略规则（已配置为只保留 data/outputs/work_dirs 目录骨架）
- .gitattributes：Git LFS 追踪规则

## 4. 目录与文件放置约定（必须遵守）

1. 新数据集先放 data/raw，再生成到 data/processed。
2. 所有实验输出统一放 outputs 或 work_dirs，不要散落在仓库根目录。
3. 配置文件统一放 configs，对应脚本通过参数读取配置，不硬编码路径。
4. 文档统一放 docs，提交前保证文档与代码路径一致。

## 5. 快速开始

1. 阅读 docs/INSTALLATION.md 完成环境搭建。
2. 小组统一搭建可参考 docs/TEAM_ENV_SETUP.md。
3. 执行 scripts/minimal_pointcloud_smoke_test.py 验证基础链路。
4. 执行 scripts/run_pointcloud_pipeline.py 生成 clean/adv/defended 样本。
5. 执行 scripts/visualize_pointcloud_samples.py 生成可视化图。
6. 执行 scripts/inspect_mvxnet_kitti_config.py 查看正式 KITTI 多模态配置清单。
7. 执行 scripts/run_mvxnet_kitti_training.py 启动正式的 KITTI 多模态训练入口。
8. 执行 scripts/run_multimodal_robustness_eval.py 进行模型级 clean/attack/defense 三分支评估。

## 6. 环境重建与提交规范（推荐）

请不要提交以下目录到 GitHub：
- .venv/
- data/ 下真实数据
- outputs/ 下真实实验产物
- work_dirs/ 下训练权重与日志大文件

推荐的正确做法：
1. 提交代码、配置、文档和环境定义文件（requirements.txt / environment.yml / Dockerfile）。
2. 通过脚本在本地或容器重建环境，而不是上传 .venv。
3. 仅保留 data/outputs/work_dirs 的目录骨架（.gitkeep/README）。

环境重建示例：
- Docker: 使用 scripts/install_ubuntu_env.sh 在容器中安装依赖。
- 本地 Python: 根据 requirements.txt 或 environment.yml 重建虚拟环境。
