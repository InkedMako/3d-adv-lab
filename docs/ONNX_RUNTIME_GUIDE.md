# ONNX Runtime 安装和使用指南

## 为什么使用ONNX Runtime？

### 优势
- ✅ **跨平台**：Windows、Linux、MacOS都能运行
- ✅ **无需编译**：预编译的二进制包，不需要CUDA Toolkit
- ✅ **GPU加速**：支持CUDA、DirectML等多种后端
- ✅ **性能优秀**：推理速度快，接近原生性能
- ✅ **易于部署**：部署简单，依赖少

### 劣势
- ⚠️ 需要先将模型转换为ONNX格式
- ⚠️ 不支持模型训练，仅支持推理
- ⚠️ 部分算子可能不支持

## 安装ONNX Runtime

### CPU版本
```powershell
pip install onnxruntime
```

### GPU版本（CUDA）
```powershell
pip install onnxruntime-gpu
```

### DirectML版本（Windows推荐，使用GPU）
```powershell
pip install onnxruntime-directml
```

## 推荐安装

对于您的RTX 4060，建议安装onnxruntime-directml，它专门为Windows GPU优化：

```powershell
pip install onnxruntime-directml
```

## 验证安装

```python
import onnxruntime as ort

# 检查可用的provider
print("可用的Providers:", ort.get_available_providers())

# 创建session
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 使用GPU（DirectML）
session = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("当前使用的Provider:", session.get_providers())
```

## 下一步

1. 安装ONNX Runtime
2. 将MMDetection3D模型转换为ONNX格式
3. 使用ONNX Runtime进行推理

## 注意事项

- ONNX Runtime仅支持推理，不支持训练
- 转换模型时需要确保所有算子都兼容
- 某些自定义操作可能需要单独处理
