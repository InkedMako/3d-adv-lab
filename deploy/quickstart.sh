#!/bin/bash
#===============================================================================
# 快速开始脚本 - 一键部署并运行实验
#===============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=============================================================================="
echo "MMDetection3D + ART 快速开始"
echo "=============================================================================="
echo ""

# 检查环境
log_info "检查系统环境..."

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "未检测到NVIDIA GPU"
    exit 1
fi

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    log_error "CUDA Toolkit未安装"
    exit 1
fi

# 显示GPU信息
log_success "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 创建conda环境
ENV_NAME="mmdet3d"
log_info "创建Python环境: $ENV_NAME"

if command -v conda &> /dev/null; then
    # 检查环境是否已存在
    if conda env list | grep -q "^$ENV_NAME "; then
        log_info "环境 $ENV_NAME 已存在"
        read -p "是否重新创建? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n $ENV_NAME -y
            conda create -n $ENV_NAME python=3.11 -y
        fi
    else
        conda create -n $ENV_NAME python=3.11 -y
    fi

    log_success "激活环境: conda activate $ENV_NAME"
    conda activate $ENV_NAME
else
    log_error "需要安装Conda才能使用此脚本"
    exit 1
fi

echo ""
log_info "安装PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

echo ""
log_info "安装MMCV..."
pip install mmcv==2.1.0

echo ""
log_info "安装MMDetection和MMDetection3D..."
pip install mmdet==3.3.0
pip install mmdet3d==1.4.0

echo ""
log_info "安装ART和其他依赖..."
pip install adversarial-robustness-toolbox
pip install numpy==1.26.4 matplotlib opencv-python pillow scipy scikit-learn pandas tqdm pyyaml

echo ""
log_info "验证安装..."
python << 'EOF'
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
except: print("  ✗ PyTorch")

try:
    import mmcv
    print(f"  ✓ MMCV {mmcv.__version__}")
except: print("  ✗ MMCV")

try:
    import mmdet3d
    print(f"  ✓ MMDetection3D {mmdet3d.__version__}")
except: print("  ✗ MMDetection3D")

try:
    import art
    print(f"  ✓ ART {art.__version__}")
except: print("  ✗ ART")
EOF

echo ""
log_success "安装完成！"
echo ""
echo "=============================================================================="
echo "运行实验"
echo "=============================================================================="
echo ""
echo "运行以下命令开始实验："
echo ""
echo "  1. 激活环境: conda activate mmdet3d"
echo "  2. 进入目录: cd ~/mmdet3d_experiments"
echo "  3. 运行实验: python scripts/run_experiment_multimodal_real.py --num-samples 10"
echo ""
echo "或者直接在这里运行："
read -p "是否现在运行实验? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "scripts" ]; then
        python scripts/run_experiment_multimodal_real.py --num-samples 10
    else
        log_error "未找到scripts目录，请确保在项目根目录"
    fi
fi
