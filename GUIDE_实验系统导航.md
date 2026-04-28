# 项目实验系统指南

本文档用于快速导航项目的实验文件、数据和可视化结果。

---

## 🗂️ 完整目录导航

```
项目根目录/
│
├── 📂 data/                          # 数据集目录
│   ├── README.md                     # ⭐ 数据组织说明
│   ├── raw/                          # 原始下载数据
│   ├── processed/                    # 处理后数据（包括 data/kitti/）
│   └── downloads/                    # 临时缓存
│
├── 📂 outputs/                       # 实验输出目录
│   ├── README.md                     # ⭐ 输出文件夹说明（含实验历史表）
│   ├── EXPERIMENTS_LOG.md            # ⭐ 中央实验日志（当前会话：烟雾测试）
│   │
│   ├── smoke_test_local/             # [实验 #1-本地] 烟雾测试
│   │   ├── clean.npy                 # 原始点云
│   │   ├── adversarial.npy           # 攻击后点云
│   │   ├── defended.npy              # 防御后点云
│   │   └── report.json               # ✅ 统计报告（文本形式）
│   │
│   ├── smoke_test/                   # [实验 #1-Docker] 烟雾测试（容器版）
│   │   ├── clean.npy/.ply
│   │   ├── adversarial.npy/.ply
│   │   ├── defended.npy/.ply
│   │   └── (无 report.json)
│   │
│   ├── vis/                          # ⭐ 可视化目录
│   │   ├── README.md                 # 可视化说明与解读指南
│   │   ├── comparison_3d.png         # ✅ 3D 散点云对比
│   │   ├── comparison_2d_xy.png      # ✅ 2D 俯视投影
│   │   └── summary.txt               # 元数据
│   │
│   └── pipeline_test/                # [预留] 完整管道实验
│
├── 📂 scripts/                       # 脚本目录
│   ├── README.md                     # 脚本说明
│   ├── minimal_pointcloud_smoke_test.py    # 烟雾测试脚本
│   └── visualize_pointcloud_samples.py     # 可视化脚本
│
├── 📂 configs/                       # 配置目录
├── 📂 docs/                          # 文档目录
└── 📂 work_dirs/                     # 模型输出目录

```

---

## 📖 快速开始：五分钟上手

### 1️⃣ 查看第一个完整实验
```bash
# 打开实验日志
notepad outputs/EXPERIMENTS_LOG.md

# 查看实验输出
ls outputs/smoke_test_local/
```

### 2️⃣ 查看可视化结果
```bash
# 打开图像
outputs/vis/comparison_3d.png           # 3D 对比
outputs/vis/comparison_2d_xy.png        # 2D 投影
```

### 3️⃣ 理解数据组织
```bash
# 查看 README
type outputs/README.md                  # 输出文件夹说明
type data/README.md                     # 数据准备说明
```

---

## 📍 关键文数据位置速查

| 需求 | 位置 | 文件 |
|------|------|------|
| **查看实验历史** | outputs/ | EXPERIMENTS_LOG.md ⭐ |
| **查看输出结构** | outputs/ | README.md ⭐ |
| **查看可视化** | outputs/vis/ | *.png ⭐ |
| **查看可视化说明** | outputs/vis/ | README.md |
| **查看数据格式** | data/ | README.md |
| **查看脚本用法** | scripts/ | README.md |
| **查看烟雾测试数据** | outputs/smoke_test_local/ | *.npy, report.json |
| **查看烟雾测试可视化** | outputs/vis/ | comparison_*.png |

---

## 🔬 实验状态速查表

### 已完成的实验

| # | 实验名 | 日期 | 数据规模 | 输出 | 可视化 | 状态 |
|---|--------|------|---------|------|--------|------|
| 1 | smoke_test_local | 2026-04-06 | 2048 点 | smoke_test_local/ | ✅ | ✅ 完成 |
| 2 | smoke_test (Docker) | 2026-04-06 | 2048 点 | smoke_test/ | ❌ | ✅ 完成 |

### 待执行的实验

| # | 实验名 | 预期日期 | 数据源 | 规模 | 目标 |
|---|--------|---------|--------|------|------|
| 3 | KITTI smoke test | TBD | data/kitti/ | 7481 | 真实数据验证 |
| 4 | 多模态融合 | TBD | data/kitti/ | 7481 | 融合检测 |

---

## 📊 实验数据对标表

### Smoke Test #1 (Local) 关键数据

```
Clean:      2048 点  (100%)      → 基准
Adversarial: 1725 点  (84.2%)    → 丢失 15.8%（符合预设 0.15）
Defended:     87 点  (4.2%)     → 压缩 95.8%（体素有效）
```

### 验证指标
- ✅ 丢点率精确度: 15.8% vs 15% (预设) → **0.8% 误差**
- ✅ 防御压缩率: 94.9% (vs 预设下采样参数)
- ✅ 统计计算: 均值、标差正确
- ✅ 文件格式: .npy 和 .ply 均可读

---

## 🚀 后续实验执行流程

### 下一步：KITTI 数据验证

1. **准备数据**
   ```bash
   # 参考 data/README.md 中的"KITTI 数据集最小下载集"
   # 下载 5 个部分到 data/raw/
   # 解压并组织到 data/kitti/
   ```

2. **创建实验目录**
   ```bash
   mkdir -p outputs/exp_2026_04_07_kitti_smoke
   ```

3. **运行管道**
   ```bash
   python scripts/run_pointcloud_pipeline.py \
     --data-root data/kitti \
     --output-dir outputs/exp_2026_04_07_kitti_smoke \
     --mode all
   ```

4. **生成可视化**
   ```bash
   python scripts/visualize_pointcloud_samples.py \
     --input-root outputs/exp_2026_04_07_kitti_smoke \
     --output-dir outputs/vis/exp_2026_04_07_kitti_smoke
   ```

5. **更新日志**
   - 编辑 `outputs/EXPERIMENTS_LOG.md`
   - 添加新的实验 #3 记录

---

## 📋 文件清单和用途

### 📄 配置和文档文件

| 文件 | 用途 | 优先级 |
|------|------|--------|
| outputs/README.md | 输出目录总体说明 | 🔴 必读 |
| outputs/EXPERIMENTS_LOG.md | 实验执行日志 | 🔴 必读 |
| outputs/vis/README.md | 可视化说明 | 🟡 推荐 |
| data/README.md | 数据集组织 | 🟡 推荐 |
| scripts/README.md | 脚本使用指南 | 🟡 推荐 |

### 🗂️ 实验数据文件

| 文件 | 格式 | 大小 | 用途 |
|------|------|------|------|
| *.npy | NumPy 数组 | 1-25 KB | 点云数据 |
| report.json | JSON | ~1 KB | 统计摘要 |
| *.ply | PLY 模型 | 50-60 KB | 三维模型格式 |

### 📊 可视化文件

| 文件 | 分辨率 | 大小 | 用途 |
|------|--------|------|------|
| comparison_3d.png | 1500x500 | 200-400 KB | 3D 对比演示 |
| comparison_2d_xy.png | 1200x400 | 150-300 KB | 2D 投影展示 |
| summary.txt | - | 1 KB | 元数据记录 |

---

## 🔗 文档互联

```
GUIDE.md (本文件)
  ├─ outputs/README.md              [输出结构详解]
  ├─ outputs/EXPERIMENTS_LOG.md      [实验记录本]
  ├─ outputs/vis/README.md           [可视化解读]
  ├─ data/README.md                  [数据准备]
  └─ scripts/README.md               [脚本使用]
```

---

## 💡 常见问题

### Q: 第一次实验的输出在哪里？
**A:** 
- 数据: `outputs/smoke_test_local/`
- 可视化: `outputs/vis/`
- 日志: `outputs/EXPERIMENTS_LOG.md`

### Q: 怎么查看实验的统计数据？
**A:** 打开 `outputs/smoke_test_local/report.json`，或参考 `outputs/EXPERIMENTS_LOG.md`

### Q: 如何添加新实验？
**A:** 
1. 创建新目录: `mkdir outputs/exp_YYYYMMDD_name/`
2. 运行实验输出到该目录
3. 在 `EXPERIMENTS_LOG.md` 中添加记录

### Q: 可视化图片在哪？
**A:** `outputs/vis/` 中，包括 3D 和 2D 对比图

### Q: KITTI 数据怎么下载和组织？
**A:** 参考 `data/README.md` 中的"KITTI 数据集最小下载集"部分

---

## ✅ 检查清单

使用此检查清单验证项目组织是否完整：

- [ ] `outputs/README.md` - 输出结构说明 ✅
- [ ] `outputs/EXPERIMENTS_LOG.md` - 实验日志 ✅
- [ ] `outputs/vis/README.md` - 可视化说明 ✅
- [ ] `outputs/smoke_test_local/` - 烟雾测试数据 ✅
- [ ] `outputs/vis/*.png` - 可视化结果 ✅
- [ ] `data/README.md` - 数据准备说明 ✅
- [ ] `scripts/README.md` - 脚本文档 ✅

---

## 📞 快速参考命令

```bash
# 查看所有实验
ls -la outputs/exp_*/

# 查看最新的输出
ls outputs/smoke_test_local/

# 生成新的可视化
python scripts/visualize_pointcloud_samples.py --input-root outputs/smoke_test_local --output-dir outputs/vis

# 验证数据完整性
python -c "import json; print(json.load(open('outputs/smoke_test_local/report.json')))"

# 查看实验日志
type outputs/EXPERIMENTS_LOG.md
```

---

## 🎯 总结

✅ **项目组织完形成** — 包含：
- 中央实验日志 (`EXPERIMENTS_LOG.md`)
- 输出目录说明 (README 系列)
- 可视化结果库 (`vis/`)
- 数据准备指南 (`data/README.md`)
- 本导航指南

**建议**：
1. 首先阅读本文件 (GUIDE.md)
2. 然后查看 `outputs/README.md`
3. 浏览 `outputs/EXPERIMENTS_LOG.md` 了解第一个实验
4. 打开可视化图片快速了解结果

