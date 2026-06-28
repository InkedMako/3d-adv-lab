# 多模态3D对抗鲁棒性研究平台

> **Multi-Modal 3D Adversarial Robustness Research Platform**

本项目是一个集点云与图像联合攻击、防御、可视化展示于一体的演示系统，基于 ART 与 MMDetection3D 构建，支持 KITTI 数据集，提供参数实时调节和实验历史管理功能。

---

## 项目简介

自动驾驶系统中的3D目标检测模型（如 MVXNet、PointPillars）面临对抗样本攻击的安全威胁。本项目旨在构建一个直观易用的演示平台，帮助理解：

- **对抗攻击**：如何在点云和图像上施加微小扰动使检测失效
- **防御方法**：如何通过滤波和去噪恢复模型的检测能力
- **参数影响**：攻击/防御参数如何影响最终效果

### 核心特性

- ✅ **多模态支持**：点云 + 图像联合攻防演示
- ✅ **实时交互**：参数滑块调节，效果即时反馈
- ✅ **全流程可视化**：原始 → 攻击 → 防御三阶段对比
- ✅ **实验历史管理**：自动记录，支持对比分析
- ✅ **KITTI数据集**：支持编号搜索和样本预览

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      前端 (React + TS)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ 配置页面 │  │ 实验页面 │  │ 历史页面 │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└───────────────────────┬─────────────────────────────────┘
                        │ RESTful API
┌───────────────────────▼─────────────────────────────────┐
│                      后端 (Flask)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  数据加载 │ 攻击模块 │ 防御模块 │ 推理 │ 可视化 │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓ 子进程调用 ↓                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Conda环境 (mm3d / MMDetection3D)         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
ed-adv/
├── backend/                 # 后端服务
│   └── app.py               # Flask 主应用
├── frontend/                # 前端应用
│   ├── src/
│   │   ├── api/             # API 调用
│   │   ├── pages/           # 页面组件
│   │   │   ├── ConfigPage.tsx    # 实验配置页
│   │   │   ├── ExperimentPage.tsx # 实验结果页
│   │   │   └ HistoryPage.tsx     # 历史记录页
│   │   └── store/           # 状态管理
│   ├── package.json
│   └── vite.config.ts
├── configs/                 # 实验配置
│   ├── base/                # 基础配置模板
│   ├── pointcloud/          # 点云任务配置
│   └── multimodal/          # 多模态任务配置
├── data/                    # 数据集
│   └ kitti/                 # KITTI 数据集
│   │   ├── training/
│   │   │   ├── velodyne/    # 点云 (.bin)
│   │   │   ├── image_2/     # 图像 (.png)
│   │   │   └ label_2/       # 标签 (.txt)
│   │   └ testing/
├── outputs/                 # 实验输出
│   ├── preview_bevs/        # BEV 预览图
│   ├── preview_images/      # 图像预览
│   └ {sample_id}_{timestamp}/  # 实验结果
│   │   ├── result.json
│   │   ├── *_comparison.png
│   │   └ *_pointcloud.png
│   └ └ experiment_scripts/     # 实验脚本缓存
├── scripts/                 # 工具脚本
│   ├── analysis/            # 分析脚本
│   ├── tools/               # 辅助工具
│   └ run_experiment_*.py    # 实验运行脚本
├── docs/                    # 项目文档
│   ├── INSTALLATION.md      # 安装指南
│   ├── WINDOWS_OPTIONS_GUIDE.md  # Windows方案选择
│   └ WSL2_CUDA_INSTALLATION.md   # WSL2 GPU配置
├── deploy/                  # 部署脚本
│   └ LINUX_DEPLOYMENT_GUIDE.md  # Linux服务器部署
├── work_dirs/               # 模型权重与日志
├── environment.yml          # Conda 环境定义
├── requirements.txt         # pip 依赖
└ Dockerfile
└ README.md
```

---

## 快速开始

### 1. 环境准备

#### 后端环境

```bash
# 创建 Conda 环境
conda env create -f environment.yml
conda activate mm3d

# 或使用 pip
pip install -r requirements.txt
```

#### 前端环境

```bash
cd frontend
npm install
```

### 2. 数据准备

下载 KITTI 数据集并放置到 `data/kitti/` 目录：

```bash
# KITTI 数据集结构
data/kitti/
├── training/
│   ├── velodyne/        # 点云数据
│   ├── image_2/         # 图像数据
│   └ label_2/           # 标注文件
│   └ calib/             # 校准文件
└── testing/
```

### 3. 启动服务

#### 启动后端

```bash
# Windows
python backend/app.py

# 或使用 Conda 环境
conda activate mm3d
python backend/app.py
```

后端服务地址：`http://localhost:5000`

#### 启动前端

```bash
cd frontend
npm run dev
```

前端服务地址：`http://localhost:5173`

### 4. 使用平台

1. **首页**：选择 KITTI 样本编号，预览点云 BEV 和图像
2. **实验页**：配置攻击/防御参数，运行实验查看效果对比
3. **历史页**：浏览历史实验记录，点击查看详情

---

## 核心功能

### 攻击算法

| 算法 | 模态 | 参数 | 说明 |
|------|------|------|------|
| **FGSM** | 点云 | epsilon, perturb_ratio | 快速梯度符号法，单步攻击 |
| **PGD** | 图像 | epsilon, steps, step_size | 投影梯度下降，迭代攻击 |

**参数说明**：
- `epsilon`：扰动强度（点云：米，图像：像素归一化值）
- `perturb_ratio`：扰动点比例（0-100%）
- `steps`：PGD迭代步数
- `step_size`：每步步长

### 防御算法

| 算法 | 模态 | 参数 | 说明 |
|------|------|------|------|
| **SOR** | 点云 | k, std_ratio | 统计离群点移除 |
| **半径滤波** | 点云 | radius, min_neighbors | 基于邻域数量去噪 |
| **高斯模糊** | 图像 | sigma, kernel_size | 平滑对抗扰动 |

**参数说明**：
- `k`：SOR近邻点数
- `std_ratio`：标准差倍数阈值
- `sigma`：高斯核标准差

### 可视化输出

- **BEV 图**：鸟瞰图视角的点云可视化
- **点云对比图**：原始/攻击/防御三阶段对比
- **图像对比图**：图像模态的攻击/防御效果
- **置信度分析**：柱状图显示各阶段置信度变化

---

## API 接口

### 样本相关

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/samples/<id>/preview` | GET | 获取样本预览（BEV、图像、标签） |
| `/api/samples/search` | POST | 搜索样本编号 |

### 实验相关

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/experiment/run` | POST | 运行实验 |
| `/api/experiment/status/<id>` | GET | 获取实验进度 |
| `/api/experiment/result/<id>` | GET | 获取实验结果 |

### 历史记录

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/history/list` | GET | 获取历史实验列表 |
| `/api/history/detail/<id>` | GET | 获取实验详情 |

### 静态文件

| 接口 | 方法 | 说明 |
|------|------|------|
| `/files/<path>` | GET | 获取输出文件（图片、报告等） |

---

## 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Flask | 2.x | Web 框架 |
| NumPy | 2.x | 数值计算 |
| OpenCV | 4.x | 图像处理 |
| PIL/Pillow | 10.x | 图像读写 |
| SciPy | 1.x | 空间计算（cKDTree） |
| MMDetection3D | 1.x | 3D检测模型 |

### 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 18.x | UI 框架 |
| TypeScript | 5.x | 类型安全 |
| Vite | 6.x | 构建工具 |
| TailwindCSS | 3.x | 样式框架 |
| Zustand | 5.x | 状态管理 |
| React Router | 7.x | 路由管理 |
| Recharts | 2.x | 图表可视化 |

### 数据集

- **KITTI Vision Benchmark Suite**
  - 点云（Velodyne LiDAR）
  - 图像（RGB Camera）
  - 3D标注（Car、Pedestrian、Cyclist等）
  - 样本数量：7481（训练集）+ 7518（测试集）

---

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10 / Ubuntu 18.04 | Windows 11 / Ubuntu 20.04 |
| Python | 3.10 | 3.10+ |
| Node.js | 18.x | 20.x+ |
| 内存 | 8 GB | 16 GB+ |
| GPU | - | NVIDIA GPU (CUDA 11.x) |
| 磁盘 | 50 GB | 100 GB+ |

---

## 开发指南

### 添加新的攻击算法

1. 在 `backend/app.py` 中添加攻击函数
2. 在前端 `ConfigPage.tsx` 中添加参数控件
3. 更新实验运行脚本

### 添加新的防御算法

1. 在 `backend/app.py` 中添加防御函数
2. 在前端参数面板中添加对应参数
3. 更新可视化生成逻辑

### 添加新的可视化

1. 在 `scripts/tools/create_visualizations.py` 中添加生成函数
2. 在前端对应页面中添加图片展示组件

---

## 文档资源

| 文档 | 路径 | 说明 |
|------|------|------|
| 安装指南 | [docs/INSTALLATION.md](docs/INSTALLATION.md) | 详细安装步骤 |
| Windows方案 | [docs/WINDOWS_OPTIONS_GUIDE.md](docs/WINDOWS_OPTIONS_GUIDE.md) | Windows开发方案选择 |
| WSL2 CUDA | [docs/WSL2_CUDA_INSTALLATION.md](docs/WSL2_CUDA_INSTALLATION.md) | WSL2 GPU配置 |
| Linux部署 | [deploy/LINUX_DEPLOYMENT_GUIDE.md](deploy/LINUX_DEPLOYMENT_GUIDE.md) | 服务器部署指南 |

---


## 常见问题

### Q: 后端启动失败？

检查：
1. Python 环境是否正确激活
2. 依赖是否完整安装
3. 端口 5000 是否被占用

### Q: 前端无法连接后端？

检查：
1. 后端服务是否正常运行
2. 前端 API 地址配置是否正确
3. CORS 配置是否生效

### Q: BEV 图无法显示？

检查：
1. 点云数据是否存在
2. KITTI 数据路径是否正确
3. outputs/preview_bevs 目录是否创建

### Q: 实验运行超时？

检查：
1. 模型推理是否需要 GPU
2. 参数设置是否合理
3. 系统资源是否充足

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 许可证

本项目仅供学术研究和教学使用，遵循相关开源协议。

---

## 致谢

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) - 3D目标检测框架
- [ART (Adversarial Robustness Toolbox)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - 对抗鲁棒性工具箱
- [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/) - 数据集提供方

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目仓库：提交 Issue
- 邮箱：[项目邮箱]

---

*最后更新：2026年6月*