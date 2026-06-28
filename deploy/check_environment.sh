#!/bin/bash
#===============================================================================
# 快速环境检查脚本 - 在部署前运行
#===============================================================================

echo "=============================================================================="
echo "MMDetection3D GPU环境检查"
echo "=============================================================================="
echo ""

# 检查1: NVIDIA GPU
echo "1. 检查NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi 可用"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "   ✗ nvidia-smi 未找到"
    echo "   请安装NVIDIA驱动"
    exit 1
fi

# 检查2: CUDA版本
echo "2. 检查CUDA版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "   ✓ nvcc 可用"
    echo "   CUDA版本: $CUDA_VERSION"

    if [[ $CUDA_VERSION == 11.7* ]] || [[ $CUDA_VERSION == 11.8* ]]; then
        echo "   ✓ CUDA版本兼容 (11.7/11.8)"
    else
        echo "   ⚠ 建议使用CUDA 11.7或11.8"
    fi
    echo ""
else
    echo "   ✗ nvcc 未找到"
    echo "   请安装CUDA Toolkit"
    exit 1
fi

# 检查3: Python版本
echo "3. 检查Python版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python版本: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc) -eq 1 ]]; then
    echo "   ✓ Python版本满足要求 (>=3.8)"
else
    echo "   ✗ Python版本过低 (需要>=3.8)"
    exit 1
fi
echo ""

# 检查4: 虚拟环境工具
echo "4. 检查虚拟环境工具..."
if command -v conda &> /dev/null; then
    echo "   ✓ Conda可用"
    conda --version
elif command -v virtualenv &> /dev/null; then
    echo "   ✓ Virtualenv可用"
elif command -v python3 &> /dev/null; then
    echo "   ✓ Python3可用 (使用venv)"
else
    echo "   ✗ 未找到虚拟环境工具"
    exit 1
fi
echo ""

# 检查5: 磁盘空间
echo "5. 检查磁盘空间..."
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
TOTAL=$(df -h . | awk 'NR==2 {print $2}')
echo "   可用空间: $AVAILABLE / $TOTAL"
if [[ $(df . | awk 'NR==2 {print $4}' | sed 's/G//') -gt 20 ]]; then
    echo "   ✓ 磁盘空间充足 (>20GB)"
else
    echo "   ⚠ 磁盘空间不足 (<20GB)"
fi
echo ""

# 总结
echo "=============================================================================="
echo "环境检查完成"
echo "=============================================================================="
echo ""
echo "✓ 您的服务器满足MMDetection3D部署要求"
echo ""
echo "下一步:"
echo "  1. 运行: bash deploy_to_linux_gpu_server.sh"
echo "  2. 或按照LINUX_DEPLOYMENT_GUIDE.md手动安装"
echo ""
