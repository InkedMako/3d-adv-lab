# Linux GPU服务器部署指南

## 概述

本文档提供在Linux GPU服务器上部署MMDetection3D + ART多模态对抗攻击防御实验的完整指南。

## 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU（至少8GB显存，建议16GB+）
- **内存**: 至少16GB RAM
- **存储**: 至少50GB可用空间

### 软件要求
- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / Rocky Linux 8+
- **CUDA**: 11.7 或 11.8
- **Python**: 3.8 - 3.11
- **NVIDIA驱动**: 470+

## 快速部署

### 方法1: 使用自动部署脚本（推荐）

```bash
# 1. 将脚本复制到服务器
scp deploy/deploy_to_linux_gpu_server.sh user@gpu-server:~/

# 2. SSH连接到服务器
ssh user@gpu-server

# 3. 给脚本添加执行权限
chmod +x deploy_to_linux_gpu_server.sh

# 4. 运行部署脚本
./deploy_to_linux_gpu_server.sh
```

### 方法2: 手动部署

#### 步骤1: 检查环境

```bash
# 检查GPU
nvidia-smi

# 检查CUDA版本
nvcc --version

# 检查Python版本
python3 --version
```

#### 步骤2: 创建虚拟环境

```bash
# 使用conda（推荐）
conda create -n mmdet3d python=3.11
conda activate mmdet3d

# 或使用venv
python3 -m venv mmdet3d_env
source mmdet3d_env/bin/activate
```

#### 步骤3: 安装PyTorch (CUDA版本)

```bash
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

#### 步骤4: 安装MMCV

```bash
pip install mmcv==2.1.0
```

#### 步骤5: 安装MMDetection和MMDetection3D

```bash
pip install mmdet==3.3.0
pip install mmdet3d==1.4.0
```

#### 步骤6: 安装ART

```bash
pip install adversarial-robustness-toolbox
```

#### 步骤7: 安装其他依赖

```bash
pip install \
    numpy==1.26.4 \
    matplotlib \
    opencv-python \
    pillow \
    scipy \
    scikit-learn \
    pandas \
    tqdm \
    pyyaml
```

## 准备项目文件

### 从Windows复制文件

```bash
# 在Windows端打开PowerShell
scp -r D:/ed-adv/* user@gpu-server:~/ed-adv/
```

### 在服务器上组织目录结构

```bash
cd ~/ed-adv

# 目录结构
# ├── configs/              # 配置文件
# ├── data/                 # 数据集
# │   └── kitti/           # KITTI数据
# ├── scripts/              # 实验脚本
# ├── outputs/              # 实验输出
# ├── backend/              # 后端服务
# ├── frontend/             # 前端应用
# └── deploy/               # 部署脚本
```

### 准备KITTI数据集

```bash
# 方法1: 软链接（如果数据在其他位置）
ln -s /path/to/kitti data/kitti

# 方法2: 直接复制
cp -r /path/to/kitti data/kitti

# 验证数据
ls -la data/kitti/training/velodyne/ | head -5
ls -la data/kitti/training/image_2/ | head -5
```

## 运行实验

### 基本运行

```bash
# 激活环境
conda activate mmdet3d

# 进入项目目录
cd ~/ed-adv

# 运行多模态实验（10个样本）
python scripts/run_experiment_multimodal_real.py --num-samples 10

# 运行不同数量的样本
python scripts/run_experiment_multimodal_real.py --num-samples 50

# 指定输出目录
python scripts/run_experiment_multimodal_real.py \
    --num-samples 20 \
    --output-dir outputs/my_experiment
```

### 调整攻击参数

```bash
# 更强的攻击
python scripts/run_experiment_multimodal_real.py \
    --num-samples 10 \
    --pc-noise-std 0.05 \
    --pc-drop-ratio 0.3 \
    --image-noise-std 0.05

# 较弱的攻击
python scripts/run_experiment_multimodal_real.py \
    --num-samples 10 \
    --pc-noise-std 0.01 \
    --pc-drop-ratio 0.1 \
    --image-noise-std 0.01
```

## 启动前后端服务

### 启动后端服务

```bash
conda activate mmdet3d
cd ~/ed-adv
python backend/app.py
```

后端服务地址：`http://localhost:5000`

### 启动前端服务

```bash
cd ~/ed-adv/frontend
npm install
npm run dev
```

前端服务地址：`http://localhost:5173`

### 反向代理配置（生产环境）

使用Nginx配置反向代理：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /files/ {
        proxy_pass http://localhost:5000/files/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 查看结果

### 实验报告

```bash
# 查看文本报告
cat outputs/experiment_multimodal_real/report.txt

# 查看JSON摘要
cat outputs/experiment_multimodal_real/summary.json | python -m json.tool
```

### 可视化结果

```bash
# 查看可视化图像
ls outputs/experiment_multimodal_real/visualizations/

# 下载到本地查看
scp user@gpu-server:~/ed-adv/outputs/experiment_multimodal_real/visualizations/*.png ./
```

## 常见问题

### 1. CUDA内存不足

```bash
# 减少batch size
python scripts/run_experiment_multimodal_real.py --num-samples 5

# 或使用较小的模型配置
```

### 2. 找不到KITTI数据

```bash
# 检查数据路径
ls -la data/kitti/training/velodyne/

# 重新创建软链接
rm -rf data/kitti
ln -s /full/path/to/kitti data/kitti
```

### 3. 依赖版本冲突

```bash
# 重新安装兼容版本
pip install numpy==1.26.4
pip install mmcv==2.1.0
pip install mmdet3d==1.4.0
```

### 4. GPU未被识别

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"

# 如果不可用，重新安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. BEV图显示为绿色

这是由于NumPy 2.2.6与matplotlib不兼容导致的。已修复为使用PIL生成BEV图：

```bash
# 删除旧的BEV文件
rm -rf outputs/preview_bevs/*.png

# 重启后端服务
python backend/app.py
```

## 性能优化

### 使用GPU加速

确保脚本使用GPU而非CPU：

```python
# 在实验脚本中设置
device = "cuda"  # 而非 "cpu"
```

### 批量处理

```bash
# 使用更多样本进行批量处理
python scripts/run_experiment_multimodal_real.py --num-samples 100
```

## 验证安装

运行以下命令验证所有组件：

```bash
python << 'EOF'
import sys
print(f"Python版本: {sys.version}\n")

components = [
    ("torch", "PyTorch"),
    ("torch.cuda", "CUDA支持"),
    ("mmcv", "MMCV"),
    ("mmdet3d", "MMDetection3D"),
    ("art", "ART"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("matplotlib", "Matplotlib"),
]

for module, name in components:
    try:
        exec(f"import {module.split('.')[0]}")
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - 未安装")

# 检查GPU
import torch
if torch.cuda.is_available():
    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n✗ GPU: 不可用")
EOF
```

## 技术支持

如果遇到问题：

1. 检查NVIDIA驱动：`nvidia-smi`
2. 检查CUDA版本：`nvcc --version`
3. 检查PyTorch CUDA支持：`python -c "import torch; print(torch.cuda.is_available())"`
4. 查看错误日志

## 下一步

- 实验完成后，查看 `outputs/experiment_multimodal_real/report.txt`
- 分析攻击成功率(ASR)和防御效果
- 根据结果调整攻击/防御参数
- 生成论文或报告所需的可视化图表
- 启动前后端服务，通过Web界面查看实验结果