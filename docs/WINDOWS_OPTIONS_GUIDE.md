# Windows上运行真实MMDetection3D模型 - 完整方案指南

## 📋 目录

1. [方案总览](#方案总览)
2. [方案一：WSL2 + CUDA（推荐⭐⭐⭐⭐⭐）](#方案一-wsl2--cuda)
3. [方案二：安装CUDA Toolkit for Windows](#方案二-安装cuda-toolkit-for-windows)
4. [方案三：使用ONNX Runtime](#方案三-使用onnx-runtime)
5. [方案四：纯CPU模拟模式](#方案四-纯cpu模拟模式)
6. [快速开始](#快速开始)

---

## 方案总览

| 方案 | 难度 | GPU支持 | 性能 | 推荐度 |
|------|------|--------|------|--------|
| **WSL2 + CUDA** | ⭐⭐ | ✅ 完整 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CUDA Toolkit | ⭐⭐⭐⭐ | ✅ 完整 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| ONNX Runtime | ⭐⭐⭐ | ⚠️ 部分 | ⭐⭐⭐ | ⭐⭐ |
| 纯CPU模拟 | ⭐ | ❌ 无 | ⭐ | ⭐⭐⭐ |

---

## 方案一：WSL2 + CUDA（推荐⭐⭐⭐⭐⭐）

### 为什么选择WSL2？

✅ **优势**
- 原生GPU加速（NVIDIA官方支持）
- 完整的Linux开发环境
- 与Windows无缝切换
- 设置简单
- 性能损失极小（<5%）

❌ **劣势**
- 需要重启启用WSL2
- 占用一定磁盘空间

### 当前状态

您的系统：
- ✅ Windows 11 Pro
- ✅ WSL2已安装
- ✅ Ubuntu 22.04已安装
- ✅ NVIDIA RTX 4060
- ✅ NVIDIA Driver 591.74

### 快速开始

**只需运行一键安装脚本：**

```powershell
# 在WSL2 Ubuntu终端中运行
bash <(curl -fsSL https://raw.githubusercontent.com/你的repo/install_wsl2.sh)
```

**或者手动安装：**

```bash
# 1. 进入WSL2
wsl

# 2. 运行安装脚本
cd ~/ed-adv/deploy
chmod +x install_wsl2.sh
./install_wsl2.sh

# 3. 复制项目文件
cp -r /mnt/d/ed-adv ~/ed-adv

# 4. 运行实验
cd ~/ed-adv
python3 scripts/run_experiment_multimodal_real.py --num-samples 10
```

详细文档：[WSL2_CUDA_INSTALLATION.md](WSL2_CUDA_INSTALLATION.md)

---

## 方案二：安装CUDA Toolkit for Windows

### 适用场景

- 不想使用WSL2
- 习惯直接在Windows开发
- 有足够的时间和耐心

### 安装步骤

1. **下载CUDA Toolkit 11.8**
   - 地址：https://developer.nvidia.com/cuda-11-8-0-download-archive
   - 大小：~2.5GB

2. **安装Visual Studio Build Tools 2022**
   - 地址：https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - 选择"使用C++的桌面开发"

3. **安装PyTorch CUDA版本**

4. **编译MMCV CUDA扩展**

### 预计时间

⏱️ **总计：1.5-2小时**

- CUDA Toolkit: 15-30分钟
- Visual Studio: 20-40分钟
- PyTorch: 10-20分钟
- MMCV编译: 30-60分钟

### 缺点

- 安装复杂
- 可能遇到各种编译错误
- 调试困难

---

## 方案三：使用ONNX Runtime

### 适用场景

- 无法安装CUDA Toolkit
- 不想使用WSL2
- 可以接受部分功能限制

### 优势

- ✅ 安装简单
- ✅ 不需要CUDA Toolkit
- ✅ 跨平台

### 劣势

- ⚠️ 需要转换模型
- ⚠️ 部分操作不支持
- ⚠️ 设置复杂

### 安装

```powershell
pip install onnxruntime-directml
```

---

## 方案四：纯CPU模拟模式

### 适用场景

- 快速验证算法
- 开题/中期演示
- 初步实验

### 当前状态

✅ **已实现并测试成功**

```powershell
python scripts/run_experiment_windows.py --num-samples 10
```

生成结果：
- 中文实验报告
- 可视化图表
- 统计数据

### 限制

- ❌ 无法使用真实MVXNet模型
- ❌ 无法获得真实识别结果
- ❌ 无法验证模型对抗鲁棒性

---

## 快速开始

### 最快方案（WSL2）

**步骤1**: 打开WSL2终端
```powershell
wsl
```

**步骤2**: 运行安装脚本
```bash
cd ~/ed-adv/deploy
./install_wsl2.sh
```

**步骤3**: 运行实验
```bash
cd ~/ed-adv
python3 scripts/run_experiment_multimodal_real.py --num-samples 10
```

**步骤4**: 查看结果
```bash
cat outputs/experiment_multimodal_real/report.txt
```

### 启动前后端服务

**启动后端**
```powershell
conda activate mm3d
python backend/app.py
```

**启动前端**
```powershell
cd frontend
npm run dev
```

**访问平台**
打开浏览器访问：`http://localhost:5173`

---

## 性能对比

| 方案 | 模型推理 | GPU加速 | 攻击效果 | 防御效果 |
|------|---------|---------|----------|----------|
| WSL2 | ✅ MVXNet | ✅ 完整 | ✅ 真实 | ✅ 真实 |
| CUDA Toolkit | ✅ MVXNet | ✅ 完整 | ✅ 真实 | ✅ 真实 |
| ONNX Runtime | ⚠️ 部分 | ⚠️ 部分 | ⚠️ 部分 | ⚠️ 部分 |
| 纯CPU | ❌ 模拟 | ❌ 无 | ⚠️ 模拟 | ⚠️ 模拟 |

---

## 推荐方案

### 1️⃣ 开题/中期演示
**推荐**: 纯CPU模拟模式
- ✅ 快速生成结果
- ✅ 完整可视化
- ✅ 可以演示算法原理

### 2️⃣ 正式实验
**推荐**: WSL2 + CUDA
- ✅ 真实模型推理
- ✅ GPU加速
- ✅ 完整功能

### 3️⃣ 论文发表
**推荐**: WSL2 + CUDA 或 Linux服务器
- ✅ 真实结果
- ✅ 可复现
- ✅ 性能最佳

---

## 常见问题

### Q1: WSL2 GPU不工作？

```bash
# 检查Windows驱动
nvidia-smi

# 更新WSL2
wsl --update
```

### Q2: 安装失败？

检查：
- 网络连接
- 磁盘空间（至少10GB）
- 管理员权限

### Q3: 内存不足？

在WSL2中限制内存：
```ini
# C:\Users\你的用户名\.wslconfig
[wsl2]
memory=8GB
processors=4
```

### Q4: 如何在Windows和WSL2之间传输文件？

```bash
# WSL2访问Windows
cd /mnt/d/ed-adv

# Windows访问WSL2
\\wsl$\Ubuntu-22.04\home\user
```

### Q5: BEV图显示为绿色？

这是由于NumPy 2.2.6与matplotlib不兼容导致的。已修复为使用PIL生成BEV图：

```powershell
# 删除旧的BEV文件
Remove-Item outputs/preview_bevs/*.png -Force

# 重启后端服务
python backend/app.py
```

---

## 下一步

1. **选择方案**：推荐WSL2
2. **运行安装**：使用一键脚本
3. **测试运行**：先运行10个样本
4. **启动服务**：启动前后端服务
5. **正式实验**：根据需要调整样本数

---

## 技术支持

如果遇到问题：

1. 查看对应方案的详细文档
2. 检查系统要求
3. 验证安装步骤
4. 搜索错误信息

---

**总结**: Windows上运行真实MMDetection3D模型**完全可以实现**，推荐使用**WSL2 + CUDA**方案，简单高效！