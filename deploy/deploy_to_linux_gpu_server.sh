#!/bin/bash
#===============================================================================
# MMDetection3D + ART 多模态对抗攻击防御实验 - Linux GPU服务器部署脚本
#===============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
if [ "$EUID" -eq 0 ]; then
    log_warning "建议不要使用root用户运行此脚本"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

#===============================================================================
# 步骤1: 系统环境检查
#===============================================================================
echo "=============================================================================="
log_info "步骤1: 检查系统环境"
echo "=============================================================================="

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU已检测到:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    log_error "未检测到NVIDIA GPU或驱动未安装"
    exit 1
fi

# 检查CUDA版本
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    log_success "CUDA版本: $CUDA_VERSION"
else
    log_error "nvcc未找到，请安装CUDA Toolkit"
    exit 1
fi

# 检查操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    log_info "操作系统: $NAME $VERSION"
fi

#===============================================================================
# 步骤2: 安装系统依赖
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤2: 安装系统依赖"
echo "=============================================================================="

# Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    log_info "使用apt-get安装系统依赖..."
    sudo apt-get update
    sudo apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3-pip \
        build-essential \
        git \
        wget \
        vim \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1

# CentOS/RHEL
elif command -v yum &> /dev/null; then
    log_info "使用yum安装系统依赖..."
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        python311 \
        python311-devel \
        python3-pip \
        git \
        wget \
        vim \
        curl \
        mesa-libGL \
        glib2 \
        libXext \
        libXrender \
        libSM

else
    log_error "不支持的包管理器"
    exit 1
fi

#===============================================================================
# 步骤3: 创建Python虚拟环境
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤3: 创建Python虚拟环境"
echo "=============================================================================="

# 检查conda是否安装
if command -v conda &> /dev/null; then
    log_info "使用Conda创建环境..."
    read -p "请输入环境名称 (默认: mmdet3d): " ENV_NAME
    ENV_NAME=${ENV_NAME:-mmdet3d}

    # 创建conda环境
    conda create -n $ENV_NAME python=3.11 -y
    conda activate $ENV_NAME

    log_success "Conda环境已创建: $ENV_NAME"

# 检查virtualenv
elif command -v virtualenv &> /dev/null; then
    log_info "使用virtualenv创建环境..."
    read -p "请输入环境路径 (默认: ~/venv/mmdet3d): " VENV_PATH
    VENV_PATH=${VENV_PATH:-~/venv/mmdet3d}

    mkdir -p $(dirname $VENV_PATH)
    virtualenv -p python3.11 $VENV_PATH
    source $VENV_PATH/bin/activate

    log_success "虚拟环境已创建: $VENV_PATH"

# 使用系统Python3
else
    log_info "使用系统Python3创建虚拟环境..."
    python3 -m venv mmdet3d_env
    source mmdet3d_env/bin/activate
    log_success "虚拟环境已创建"
fi

#===============================================================================
# 步骤4: 升级pip和基础工具
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤4: 升级pip和基础工具"
echo "=============================================================================="

pip install --upgrade pip setuptools wheel

#===============================================================================
# 步骤5: 安装PyTorch (CUDA版本)
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤5: 安装PyTorch (CUDA版本)"
echo "=============================================================================="

# 根据CUDA版本选择合适的PyTorch
if [[ $CUDA_VERSION == 11.8* ]]; then
    log_info "安装PyTorch for CUDA 11.8..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118
elif [[ $CUDA_VERSION == 11.7* ]]; then
    log_info "安装PyTorch for CUDA 11.7..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu117
else
    log_warning "检测到CUDA $CUDA_VERSION，使用默认CUDA版本..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118
fi

# 验证PyTorch安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

#===============================================================================
# 步骤6: 安装MMCV
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤6: 安装MMCV"
echo "=============================================================================="

pip install mmcv==2.1.0

# 验证MMCV安装
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"

#===============================================================================
# 步骤7: 安装MMDetection和MMDetection3D
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤7: 安装MMDetection和MMDetection3D"
echo "=============================================================================="

# 安装mmdet
pip install mmdet==3.3.0

# 安装mmdet3d
pip install mmdet3d==1.4.0

# 验证安装
python -c "import mmdet3d; print(f'MMDetection3D: {mmdet3d.__version__}')"

#===============================================================================
# 步骤8: 安装ART (Adversarial Robustness Toolbox)
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤8: 安装ART"
echo "=============================================================================="

pip install adversarial-robustness-toolbox

# 验证ART安装
python -c "import art; print(f'ART: {art.__version__}')"

#===============================================================================
# 步骤9: 安装其他依赖
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤9: 安装其他依赖"
echo "=============================================================================="

pip install \
    numpy==1.26.4 \
    matplotlib \
    opencv-python \
    pillow \
    scipy \
    scikit-learn \
    pandas \
    tqdm \
    pyyaml \
    tensorboard \
    pytest

#===============================================================================
# 步骤10: 准备数据目录
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤10: 准备数据目录"
echo "=============================================================================="

# 创建项目目录
PROJECT_DIR=~/mmdet3d_experiments
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

log_info "项目目录: $PROJECT_DIR"

# 复制或克隆项目文件
if [ -d "/mnt/c/ed-adv" ] || [ -d "d:/ed-adv" ]; then
    log_warning "检测到Windows项目目录，请手动复制项目文件到服务器"
    log_info "或者使用scp命令: scp -r user@windows-pc:/path/to/project/* ."
fi

# 创建数据软链接（如果数据已存在）
if [ -d "/data/kitti" ]; then
    ln -sf /data/kitti data/kitti
    log_success "KITTI数据集软链接已创建"
elif [ -d "/mnt/data/kitti" ]; then
    ln -sf /mnt/data/kitti data/kitti
    log_success "KITTI数据集软链接已创建"
else
    log_warning "请手动下载或链接KITTI数据集"
    log_info "数据集应放在 data/kitti 目录下"
fi

#===============================================================================
# 步骤11: 验证完整安装
#===============================================================================
echo ""
echo "=============================================================================="
log_info "步骤11: 验证完整安装"
echo "=============================================================================="

python << 'EOF'
import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import mmcv
    print(f"✓ MMCV: {mmcv.__version__}")
except ImportError as e:
    print(f"✗ MMCV: {e}")

try:
    import mmdet3d
    print(f"✓ MMDetection3D: {mmdet3d.__version__}")
except ImportError as e:
    print(f"✗ MMDetection3D: {e}")

try:
    import art
    print(f"✓ ART: {art.__version__}")
except ImportError as e:
    print(f"✗ ART: {e}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")

print("\n所有依赖安装完成！")
EOF

#===============================================================================
# 完成
#===============================================================================
echo ""
echo "=============================================================================="
log_success "部署完成！"
echo "=============================================================================="

echo ""
log_info "下一步操作:"
echo ""
echo "1. 激活虚拟环境:"
if command -v conda &> /dev/null; then
    echo "   conda activate $ENV_NAME"
else
    echo "   source mmdet3d_env/bin/activate"
fi
echo ""
echo "2. 复制项目文件到服务器:"
echo "   scp -r user@windows-pc:/d/ed-adv/* user@gpu-server:~/mmdet3d_experiments/"
echo ""
echo "3. 运行实验:"
echo "   cd ~/mmdet3d_experiments"
echo "   python scripts/run_experiment_multimodal_real.py --num-samples 10"
echo ""
echo "4. 查看结果:"
echo "   cat outputs/experiment_multimodal_real/report.txt"
echo ""
echo "=============================================================================="
