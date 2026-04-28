# Data

Use this directory as the dataset staging area.

## Suggested structure

- `raw/`: downloaded original datasets
- `processed/`: converted or preprocessed dataset artifacts
- `downloads/`: temporary download cache

Do not hardcode dataset paths in code. Use relative paths from this directory whenever possible.

---

## 📁 目录结构详细说明

### 1. **raw/** [原始下载数据]
用于存放从官方 or 镜像源下载的原始数据集（未处理）。

#### 下载文件清单示例
```
data/raw/
├── KITTI_download_notes.md       # 下载说明和源链接
├── left_color_images.zip         # 左摄像头 RGB 图像
├── velodyne_points.tar.gz        # Velodyne 点云激光雷达数据
├── calib.tar.gz                  # 标定文件（内参/外参）
├── label_2.zip                   # 2D/3D 标签（bboxes）
└── devkit_object.tar.gz          # 官方评测工具
```

### 2. **kitti/** [项目使用的数据根目录]
项目中的 KITTI 配置默认读取 `data/kitti/`。

#### KITTI 标准目录结构
```
data/kitti/
├── README.md                     # KITTI 数据集说明
├── ImageSets/
│   ├── train.txt                # 训练集索引（样本编号列表）
│   ├── val.txt                  # 验证集索引
│   └── test.txt                 # 测试集索引
├── training/                    # 训练集数据
│   ├── image_2/                 # 左摄像头 RGB 图像
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── velodyne/                # Velodyne 点云 (.bin 格式)
│   │   ├── 000000.bin
│   │   ├── 000001.bin
│   │   └── ...
│   ├── calib/                   # 标定参数
│   │   ├── 000000.txt           # 内参 K, 外参 Tr_velo_to_cam
│   │   ├── 000001.txt
│   │   └── ...
│   └── label_2/                 # 3D 标签
│       ├── 000000.txt
│       ├── 000001.txt
│       └── ...
├── testing/                     # 测试集数据
│   ├── image_2/
│   ├── velodyne/
│   └── calib/
└── devkit_object/               # 官方评测工具
		├── cpp/
		├── python/
		└── README.md
```

### 3. **downloads/** [临时缓存]
用于存放正在下载或临时处理的文件（`.gitignore` 忽略此目录）。

---

## 📥 KITTI 数据集最小下载集

### 必需的 5 个部分

| 部分 | 文件名 | 用途 | 大小 |
|------|--------|------|-----|
| **左彩色图像** | left_color_images.zip | 多模态输入（RGB） | ~5 GB |
| **Velodyne 点云** | velodyne_points.tar.gz | 多模态输入（LiDAR） | ~5 GB |
| **标定文件** | calib.tar.gz | 坐标对齐（相机↔激光） | ~10 MB |
| **3D 标签** | label_2.zip | 训练监督信号 | ~20 MB |
| **开发工具** | devkit_object.tar.gz | 评测 mAP 等指标 | ~5 MB |

### 下载步骤

1. **访问官方网站**: http://www.cvlibs.net/datasets/kitti/eval_object.php
2. **创建账户并同意协议**
3. **下载上述 5 个文件** 到 `data/raw/`
4. **解压并组织** 到 `data/kitti/`

### 脚本化下载示例（可选）
```bash
# 使用 dataset 工具（需要 KITTI 镜像源配置）
cd data/raw
# 调用即将编写的下载脚本 scripts/download_kitti.py
python ../../scripts/download_kitti.py \
	--output-dir . \
	--parts all  # 或指定：images,velodyne,calib,labels
```

---

## 🔄 数据准备检查清单

使用此清单验证数据是否正确准备：

- [ ] `data/kitti/ImageSets/train.txt` 存在且包含样本索引
- [ ] `data/kitti/training/image_2/000000.png` 等图像文件存在
- [ ] `data/kitti/training/velodyne/000000.bin` 等点云文件存在
- [ ] `data/kitti/training/calib/000000.txt` 等标定文件存在
- [ ] `data/kitti/training/label_2/000000.txt` 等标签文件存在
- [ ] `data/kitti/devkit_object/` 目录包含评测脚本

验证命令：
```bash
# 检查文件数目
ls data/kitti/training/image_2/ | wc -l    # 应该 ≈ 7481
ls data/kitti/training/velodyne/ | wc -l   # 应该 ≈ 7481
```

---

## 📝 使用规范

### ✅ 推荐做法
```python
# 使用相对路径
from pathlib import Path
KITTI_ROOT = Path(__file__).parent.parent / "data" / "kitti"
train_images = KITTI_ROOT / "training" / "image_2"
```

### ❌ 避免硬编码
```python
# 不要这样写
train_images = "/home/user/project/data/kitti/training/image_2"
```

---

## 🔗 参考资源

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- [3D Object Detection Evaluation Pages](http://www.cvlibs.net/datasets/kitti/eval_object.php)
- [论文](https://arxiv.org/pdf/1504.00325.pdf) - "Are we ready for autonomous driving?"
