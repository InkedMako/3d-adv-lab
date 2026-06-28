# Windows CUDA Toolkit 安装指南

## 概述

本文档说明如何在Windows上安装CUDA Toolkit，以支持MMDetection3D的CUDA扩展编译。

## 当前状态

您的系统：
- ✅ NVIDIA GPU: RTX 4060 Laptop (8GB)
- ✅ NVIDIA Driver: 591.74
- ✅ CUDA Driver Version: 13.1
- ❌ CUDA Toolkit: 未安装
- ❌ nvcc编译器: 不可用

## 安装CUDA Toolkit

### 步骤1: 下载CUDA Toolkit

访问NVIDIA官网下载CUDA Toolkit 11.8：

**下载地址**: https://developer.nvidia.com/cuda-11-8-0-download-archive

或者直接访问：
```
https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_windows.exe
```

### 步骤2: 运行安装程序

1. 双击下载的`.exe`文件
2. 选择"Custom (Advanced)"安装选项
3. 确保选择以下组件：
   - ✅ CUDA Toolkit
   - ✅ CUDA Runtime
   - ✅ CUDA Documentation
   - ✅ CUDA Samples (可选)
   - ✅ CUDA Tools (可选)

4. **注意**：取消选择"Visual Studio Integration"（如果不需要）

### 步骤3: 验证安装

安装完成后，打开新的PowerShell窗口，运行：

```powershell
nvcc --version
```

应该显示：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on win_amd64
Cuda compilation tools, release 11.8, V11.8.91
Build cuda_11.8.r11.8/compiler.31916440_1
```

### 步骤4: 配置环境变量

CUDA Toolkit安装程序通常会自动配置环境变量。如果没有，请手动添加：

**系统变量**：
```
变量名: CUDA_PATH
变量值: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

**Path变量添加**：
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
```

### 步骤5: 安装Visual Studio Build Tools（编译必需）

MMCV的CUDA扩展需要C++编译器编译。安装Visual Studio Build Tools：

1. 下载 Visual Studio Build Tools 2022：
   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

2. 运行安装程序

3. 选择"使用C++的桌面开发"工作负载

4. 确保安装：
   - ✅ MSVC v143 - VS 2022 C++ x64/x86 生成工具
   - ✅ Windows 11 SDK (或Windows 10 SDK)

### 步骤6: 重新安装PyTorch和MMCV

完成以上步骤后，重新安装PyTorch和MMCV：

```powershell
# 1. 卸载旧版本
pip uninstall torch torchvision torchaudio mmcv mmdet mmdet3d -y

# 2. 安装PyTorch CUDA版本
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 设置编译环境变量
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"

# 4. 安装MMCV（会自动编译CUDA扩展）
pip install mmcv==2.1.0

# 5. 安装MMDetection
pip install mmdet==3.3.0

# 6. 安装MMDetection3D
pip install mmdet3d==1.4.0
```

### 步骤7: 验证安装

运行以下命令验证：

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"

python -c "import mmdet3d; print(f'MMDetection3D: {mmdet3d.__version__}')"
```

## 常见问题

### Q1: 安装后nvcc找不到？

**A**: 确保：
1. 重启PowerShell/终端（环境变量需要更新）
2. 检查CUDA_PATH环境变量
3. 手动添加CUDA bin到PATH

### Q2: MMCV编译失败？

**A**: 确保：
1. 已安装Visual Studio Build Tools
2. 已安装Windows SDK
3. 设置了CUDA_HOME环境变量
4. 使用正确的MMCV版本

### Q3: 编译时内存不足？

**A**: MMCV CUDA扩展编译需要大量内存：
1. 关闭其他程序
2. 增加Windows虚拟内存
3. 使用Release模式编译

### Q4: 找不到CUDA头文件？

**A**: 检查CUDA_PATH设置：
```powershell
echo $env:CUDA_PATH
dir "$env:CUDA_PATH\include\cuda.h"
```

## 替代方案：如果不想安装完整CUDA Toolkit

### 方案A：使用WSL2 + CUDA（推荐）

如果您的系统支持WSL2，这是更好的选择：

1. 启用WSL2
2. 在WSL2中安装Ubuntu
3. 在WSL2中安装完整的CUDA Toolkit
4. 在WSL2中运行MMDetection3D

优点：Linux环境对CUDA支持更好

### 方案B：使用预编译的MMCV

查看是否有预编译的MMCV wheel：

https://download.openmmlab.com/mmcv/dist/

查找：
- torch2.1
- cu118
- win_amd64
- cp311

如果有，可以直接安装，跳过编译步骤。

## 预计安装时间

- CUDA Toolkit: 15-30分钟
- Visual Studio Build Tools: 20-40分钟
- PyTorch + MMCV安装: 30-60分钟
- **总计**: 约1.5-2小时

## 完成后

安装成功后，您可以：

✅ 使用真实MVXNet模型进行多模态推理
✅ GPU加速推理
✅ 完整的对抗攻击防御实验
✅ 获得真实模型识别结果

## 技术支持

如果遇到问题：

1. 查看NVIDIA安装日志
2. 检查Visual Studio安装日志
3. 验证环境变量配置
4. 搜索错误信息
