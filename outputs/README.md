# Outputs

Use this directory for attack samples, defended samples, visualizations, and evaluation reports.

Current smoke test outputs are stored under `outputs/smoke_test/`.

---

## 📁 实验文件夹详细说明

### 1. **smoke_test/** [Docker 环境首次验证]
- **执行时间**: 2026-04-06
- **环境**: Docker 容器（3d-adv-lab:cu124）
- **点云规模**: 2048 个随机点
- **内容**:
	- `clean.npy` (24.7 KB) - 原始点云
	- `adversarial.npy` (20.8 KB) - 攻击后（Gaussian noise std=0.02, drop_ratio=15%）
	- `defended.npy` (20.8 KB) - 防御后（Voxel downsampling voxel_size=0.05）
	- `*.ply` - 三维模型格式副本
- **备注**: 无 report.json（默认无头模式）

### 2. **smoke_test_local/** [本地环境验证，推荐参考]
- **执行时间**: 2026-04-06
- **环境**: Windows 本地 Python（无 GPU 依赖）
- **点云规模**: 2048 个随机点
- **内容**:
	- `clean.npy` - 原始点云
	- `adversarial.npy` - 攻击后点云
	- `defended.npy` - 防御后点云
	- `report.json` ⭐ - 统计报告（点数、均值、标差、攻防参数）
- **用途**: 验证核心算法逻辑（不依赖 CUDA/mmdet3d）
- **对应可视化**: `vis/comparison_*.png`

### 3. **vis/** [可视化输出目录]
- **来源**: visualization 脚本处理 smoke_test_local
- **内容**:
	- `comparison_3d.png` ⭐ - 三维散点对比图
		- Clean：蓝色散点（2048个）
		- Adversarial：红色散点（1725个，丢失15.8%）
		- Defended：绿色散点（87个，仅保留4.2%）
	- `comparison_2d_xy.png` - XY 平面投影图（俯视）
	- `summary.txt` - 形状信息摘要
- **生成命令**:
	```bash
	python scripts/visualize_pointcloud_samples.py \
		--input-root outputs/smoke_test_local \
		--output-dir outputs/vis --dpi 150
	```

### 4. **pipeline_test/** [待用，完整管道预留]
- **用途**: 完整管道实验（使用 KITTI 或其他真实数据集）
- **状态**: 目录骨架已创建，等待数据
- **预期内容**: 多样本、多模态、完整评估指标

---

## 📊 实验历史表

| 日期 | 实验名 | 环境 | 规模 | 输出格式 | 可视化 | 状态 |
|------|--------|------|------|---------|--------|------|
| 2026-04-06 | smoke_test | Docker | 2048点 | .npy/.ply | ❌ | ✅ |
| 2026-04-06 | smoke_test_local | Python | 2048点 | .npy/.json | ✅ | ✅ |

---

## 🔄 添加新实验的步骤

**示例：执行 KITTI 烟雾测试**

1. 创建新实验文件夹（推荐格式: `exp_YYYY_MM_DD_实验名`）
	 ```bash
	 mkdir -p outputs/exp_2026_04_07_kitti_smoke
	 ```

2. 运行管道，输出到该文件夹
	 ```bash
	 python scripts/run_pointcloud_pipeline.py \
		 --data-root data/kitti \
		 --output-dir outputs/exp_2026_04_07_kitti_smoke \
		 --mode all
	 ```

3. 生成对应的可视化（可选）
	 ```bash
	 python scripts/visualize_pointcloud_samples.py \
		 --input-root outputs/exp_2026_04_07_kitti_smoke \
		 --output-dir outputs/vis/exp_2026_04_07_kitti_smoke \
		 --dpi 200
	 ```

4. 更新本 README 以记录新实验

*** End Patch
