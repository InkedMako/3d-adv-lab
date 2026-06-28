#!/bin/bash
#===============================================================================
# WSL2 Ubuntu MMDetection3D 一键安装脚本
#===============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=============================================================================="
echo "WSL2 Ubuntu MMDetection3D 环境安装"
echo "=============================================================================="
echo ""

# 1. 检查GPU
echo -e "${BLUE}[1/8]${NC} 检查GPU支持..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} GPU检测到:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}✗${NC} 未检测到GPU"
    exit 1
fi

# 2. 检查CUDA
echo ""
echo -e "${BLUE}[2/8]${NC} 检查CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo -e "${GREEN}✓${NC} CUDA版本: $CUDA_VERSION"
else
    echo -e "${YELLOW}!${NC} CUDA Toolkit未安装，开始安装..."
    echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /" | \
        sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-11-8
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    echo -e "${GREEN}✓${NC} CUDA Toolkit安装完成"
fi

# 3. 更新系统
echo ""
echo -e "${BLUE}[3/8]${NC} 更新系统..."
sudo apt update
sudo apt upgrade -y

# 4. 安装Python和依赖
echo ""
echo -e "${BLUE}[4/8]${NC} 安装Python和基础依赖..."
sudo apt install -y python3.11 python3.11-dev python3-pip build-essential git curl vim

# 5. 安装基础Python库
echo ""
echo -e "${BLUE}[5/8]${NC} 安装Python基础库..."
pip3 install --upgrade pip setuptools wheel
pip3 install numpy==1.26.4 matplotlib opencv-python pillow scipy scikit-learn pandas tqdm pyyaml

# 6. 安装PyTorch (CUDA版本)
echo ""
echo -e "${BLUE}[6/8]${NC} 安装PyTorch..."
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证PyTorch
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA可用: {torch.cuda.is_available()}')"

# 7. 安装MMCV
echo ""
echo -e "${BLUE}[7/8]${NC} 安装MMCV..."
pip3 install openmim
mim install mmcv==2.1.0

# 验证MMCV
python3 -c "import mmcv; print(f'  MMCV: {mmcv.__version__}')"

# 8. 安装MMDetection和MMDetection3D
echo ""
echo -e "${BLUE}[8/8]${NC} 安装MMDetection和MMDetection3D..."
pip3 install mmdet==3.3.0
pip3 install mmdet3d==1.4.0

# 安装ART
pip3 install adversarial-robustness-toolbox

# 验证安装
echo ""
echo "=============================================================================="
echo "验证安装"
echo "=============================================================================="
python3 << 'EOF'
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

print("\n所有依赖安装完成！")
EOF

# 完成
echo ""
echo "=============================================================================="
echo -e "${GREEN}✓ 安装完成！${NC}"
echo "=============================================================================="
echo ""
echo "下一步操作:"
echo ""
echo "  1. 复制项目文件到WSL2:"
echo "     cp -r /mnt/d/ed-adv ~/mmdet3d_experiments"
echo ""
echo "  2. 进入项目目录:"
echo "     cd ~/mmdet3d_experiments"
echo ""
echo "  3. 运行实验:"
echo "     python3 scripts/run_experiment_multimodal_real.py --num-samples 10"
echo ""
echo "  4. 查看结果:"
echo "     cat outputs/experiment_multimodal_real/report.txt"
echo ""
