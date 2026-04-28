# 可视化结果库 (Visualization Outputs)

本目录存放所有实验的可视化结果（图表、图像等）。

---

## 📁 当前文件清单

### smoke_test_local 烟雾测试可视化

#### 📊 comparison_3d.png
**三维散点云对比图**

- **分辨率**: 1500x500 (3 个 500x500 子图)
- **内容**: 三个 3D 坐标系并排显示
  - **左**: Clean (蓝色) - 原始 2048 随机点
  - **中**: Adversarial (红色) - 攻击后 1725 点 (丢失 15.8%)
  - **右**: Defended (绿色) - 防御后 87 点 (仅保留 4.2%)

**关键观察**:
```
Clean:        完整的球形分布（半径 ≈ 3）
Adversarial:  轮廓保持，局部噪声扰动可见
Defended:     极端稀疏化，仅保留代表性簇
```

#### 📊 comparison_2d_xy.png
**二维俯视投影图**

- **分辨率**: 1200x400 (3 个 400x400 子图)
- **投影**: XY 平面（顶视图）
- **用途**: 观察水平分布，更容易看到丢点和聚类效果
- **坐标范围**: [-4, 3] x [-4, 3]

**关键观察**:
```
Clean vs Adversarial:   覆盖范围相似，密度略有差异
Adversarial vs Defended: 从连续到离散的剧烈变化
```

#### 📄 summary.txt
**元数据摘要**

内容: 各点云的形状信息
```
clean: (2048, 3)        # 2048 个 3D 点
adversarial: (1725, 3)  # 1725 个 3D 点
defended: (87, 3)       # 87 个 3D 点
```

---

## 🔄 如何生成这些可视化

### 从 smoke_test_local 生成
```bash
python scripts/visualize_pointcloud_samples.py \
  --input-root outputs/smoke_test_local \
  --output-dir outputs/vis \
  --dpi 150
```

### 自定义参数
```bash
# 提高分辨率
python scripts/visualize_pointcloud_samples.py \
  --input-root outputs/smoke_test_local \
  --output-dir outputs/vis \
  --dpi 300  # 300 DPI 输出更清晰

# 从 Docker 实验生成
python scripts/visualize_pointcloud_samples.py \
  --input-root outputs/smoke_test \
  --output-dir outputs/vis/smoke_test_docker \
  --dpi 150
```

---

## 📐 可视化脚本工作原理

### 输入要求
脚本期望输入目录包含:
- `clean.npy` - 原始点云
- `adversarial.npy` - 攻击后点云  
- `defended.npy` - 防御后点云

### 生成过程
1. **加载 .npy 文件** 转换为 numpy 数组
2. **创建 3D 散点图**:
   - Matplotlib Figure (1 行 3 列子图)
   - 每个子图对应一种点云
   - 使用不同颜色区分
   - 自动计算点数并标注
3. **创建 2D 投影**:
   - 提取 X, Y 坐标
   - 绘制 2D 散点图
4. **保存文件**:
   - PNG 格式 (配置 DPI)
   - TXT 文本摘要

### 代码位置
[scripts/visualize_pointcloud_samples.py](../scripts/visualize_pointcloud_samples.py)

---

## 🎨 颜色编码约定

为了保持一致性，所有可视化使用以下颜色方案：

| 类型 | 颜色 | RGB |
|------|------|-----|
| Clean（清洁） | 蓝色 | #1f77b4 |
| Adversarial（对抗） | 红色 | #ff7f0e |
| Defended（防御） | 绿色 | #2ca02c |

---

## 📊 解读指南

### 什么表示攻击有效？
- ✅ Adversarial 丢失部分点（不是完全相同）
- ✅ Adversarial 仍保持原始形状（不是完全破坏）
- ✅ 噪声虽小但可见（对比密度/分布）

### 什么表示防御有效？
- ✅ Defended 显著稀疏（点大幅减少）
- ✅ Defended 仍覆盖原始空间（不是完全消失）
- ✅ 数据压缩效率高（减少噪声干扰余地）

---

## 🔗 后续可视化计划

### 第二阶段（使用 KITTI 数据）
- [ ] 实际扫描的点云分布（非随机）
- [ ] 多个对象的 3D bounding boxes
- [ ] 图像 + 点云融合视图
- [ ] 攻防效果的定量图表

### 第三阶段（多模态）
- [ ] 点云投影到图像平面
- [ ] 联合的检测结果对比
- [ ] 攻击成功率曲线 (AP 指标)

---

## 📝 添加新可视化的步骤

1. **在脚本中新增函数**（如果需要）
   ```python
   # 示例: 添加直方图
   def plot_distance_histogram(points, ax, title):
       distances = np.linalg.norm(points, axis=1)
       ax.hist(distances, bins=50, alpha=0.7)
       ax.set_title(title)
   ```

2. **运行可视化脚本生成新图**
   ```bash
   python scripts/visualize_pointcloud_samples.py \
     --input-root outputs/smoke_test_local \
     --output-dir outputs/vis --dpi 150
   ```

3. **验证输出**
   ```bash
   ls outputs/vis/
   ```

4. **更新本 README** 记录新产出

---

## 📸 快速浏览

| 文件 | 用途 | 最适合 |
|------|------|--------|
| comparison_3d.png | 3D 全景对比 | 报告、演示 |
| comparison_2d_xy.png | 2D 俯视对比 | 论文图表 |
| summary.txt | 基本数据 | 记录、验证 |

---

## 💾 文件大小参考

```
comparison_3d.png    ≈ 200-400 KB (取决于 DPI)
comparison_2d_xy.png ≈ 150-300 KB (取决于 DPI)
summary.txt          ≈ 1 KB
```

---

## 🔗 相关文档

- [实验日志](./EXPERIMENTS_LOG.md) - 所有实验记录
- [输出说明](./README.md) - 目录结构
- [脚本说明](../scripts/README.md) - 可视化脚本细节
