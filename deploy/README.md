# GPU服务器部署方案

## 概述

由于MVXNet等多模态3D检测模型需要CUDA GPU支持才能运行真实推理，本文档提供将项目部署到Linux GPU服务器的完整方案。

## 问题说明

### 当前环境限制
- **操作系统**: Windows
- **GPU**: NVIDIA RTX 4060 Laptop (检测到但未完全利用)
- **问题**: Windows系统缺少CUDA Toolkit开发工具链（nvcc编译器）
- **影响**: 无法编译MMCV的CUDA扩展模块
- **结果**: MMDetection3D只能运行在CPU模式，无法使用真实模型推理

### MVXNet模型需求
MVXNet使用动态点云体素化操作（`dynamic_point_to_voxel_forward`），这些CUDA操作：
- 需要nvcc编译
- 仅支持GPU执行
- CPU模式下会报错

## 解决方案

### 推荐方案：使用Linux GPU服务器

**优势：**
- ✅ 完整的CUDA Toolkit支持
- ✅ GPU加速推理
- ✅ 更好的深度学习框架兼容性
- ✅ 稳定的运行环境

**服务器要求：**
- NVIDIA GPU（至少8GB显存，推荐16GB+）
- CUDA 11.7 或 11.8
- Ubuntu 18.04+ / CentOS 7+
- NVIDIA驱动 470+

## 部署文件说明

本目录包含以下部署文件：

### 1. `deploy_to_linux_gpu_server.sh` ⭐推荐
**功能**: 全自动部署脚本
- 检查系统环境（GPU、CUDA、Python）
- 创建Python虚拟环境（支持conda/venv）
- 安装PyTorch CUDA版本
- 安装MMCV、MMDetection、MMDetection3D
- 安装ART和其他依赖
- 验证安装完整性

**使用方式:**
```bash
# 在Linux服务器上运行
chmod +x deploy_to_linux_gpu_server.sh
./deploy_to_linux_gpu_server.sh
```

### 2. `quickstart.sh`
**功能**: 快速开始脚本
- 一键安装所有依赖
- 交互式引导
- 自动验证安装

**使用方式:**
```bash
chmod +x quickstart.sh
./quickstart.sh
```

### 3. `check_environment.sh`
**功能**: 环境预检查脚本
- 验证GPU可用性
- 检查CUDA版本
- 检查Python版本
- 验证磁盘空间

**使用方式:**
```bash
chmod +x check_environment.sh
./check_environment.sh
```

### 4. `LINUX_DEPLOYMENT_GUIDE.md`
**功能**: 详细部署文档
- 完整的安装步骤
- 常见问题解答
- 性能优化建议
- 验证和测试方法

### 5. `transfer_to_server.ps1`
**功能**: Windows文件传输脚本
- 交互式传输配置
- 使用rsync/scp传输文件
- 自动排除大文件和缓存

**使用方式:**
```powershell
# 在Windows PowerShell中运行
.\transfer_to_server.ps1
```

## 部署步骤

### 步骤1: 环境预检查

首先在目标Linux服务器上运行环境检查：

```bash
# SSH到服务器
ssh user@gpu-server

# 运行检查脚本
bash ~/path/to/check_environment.sh
```

### 步骤2: 传输项目文件

**方式A: 使用PowerShell脚本（推荐）**

在Windows PowerShell中：
```powershell
.\deploy\transfer_to_server.ps1
```

**方式B: 手动使用scp**

```bash
# 在Windows PowerShell中
scp -r D:/ed-adv/* user@gpu-server:~/mmdet3d_experiments/
```

### 步骤3: 运行部署脚本

SSH到服务器：

```bash
ssh user@gpu-server
cd ~/mmdet3d_experiments
bash deploy/deploy_to_linux_gpu_server.sh
```

### 步骤4: 验证并运行实验

```bash
# 激活环境
conda activate mmdet3d

# 运行实验
python scripts/run_experiment_multimodal_real.py --num-samples 10

# 查看结果
cat outputs/experiment_multimodal_real/report.txt
```

## 替代方案

### 方案A: Windows WSL2 + CUDA

如果不想使用远程服务器，可以考虑：

1. 安装WSL2
2. 在WSL2中安装NVIDIA驱动和CUDA Toolkit
3. 在WSL2中运行部署脚本

**优点**: 无需额外服务器
**缺点**: WSL2对GPU支持有限，可能有性能损失

### 方案B: 云GPU服务器

使用云服务提供商的GPU实例：

- **AWS EC2** (p3/p4实例)
- **Google Cloud** (GPU实例)
- **阿里云** (GN系列实例)
- **腾讯云** (GPU实例)

**优点**: 即开即用，无需管理硬件
**缺点**: 需要付费

### 方案C: 使用纯CPU模型

如果GPU实在不可用：

1. 修改实验脚本使用模拟预测模式
2. 专注于攻击防御算法验证
3. 不进行真实模型推理

**优点**: 无需GPU
**缺点**: 无法验证真实模型对抗鲁棒性

## 常见问题

### Q1: 没有GPU服务器怎么办？

**A**: 可以使用云GPU服务器或纯CPU模拟模式。

### Q2: 公司/学校服务器无法安装软件怎么办？

**A**: 联系管理员说明需求，或使用conda环境在用户目录下安装。

### Q3: 数据如何传输到服务器？

**A**: 使用scp、rsync或SFTP传输，参考`transfer_to_server.ps1`。

### Q4: 部署脚本运行失败怎么办？

**A**:
1. 查看错误信息
2. 运行`check_environment.sh`检查环境
3. 参考`LINUX_DEPLOYMENT_GUIDE.md`手动安装

### Q5: 如何确认GPU推理是否正常工作？

**A**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from mmdet3d.apis import init_model; m = init_model('configs/multimodal/mvxnet_kitti_3class.py', device='cuda'); print('模型加载成功')"
```

## 技术支持

如果遇到问题：

1. **检查GPU**: `nvidia-smi`
2. **检查CUDA**: `nvcc --version`
3. **检查PyTorch**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **查看日志**: 检查错误输出的具体信息
5. **搜索引擎**: 搜索错误信息寻找解决方案

## 预期结果

成功部署后，您将能够：

- ✅ 使用真实MVXNet模型进行点云+图像多模态推理
- ✅ 运行对抗攻击实验（点云噪声、点丢弃、图像噪声）
- ✅ 运行对抗防御实验（体素下采样、图像去噪）
- ✅ 生成详细的实验报告和可视化结果
- ✅ 获得真实的模型识别准确率、攻击成功率、防御效果数据

## 总结

| 项目 | 当前Windows环境 | Linux GPU服务器 |
|------|----------------|----------------|
| GPU支持 | ⚠️ 部分 | ✅ 完整 |
| CUDA Toolkit | ❌ 缺失 | ✅ 完整 |
| MVXNet推理 | ❌ 仅模拟 | ✅ 真实 |
| 推理速度 | 慢(CPU) | 快(GPU) |
| 实用性 | 研究/调试 | 正式实验 |

**推荐**: 使用Linux GPU服务器进行正式实验，Windows环境用于代码开发和调试。
