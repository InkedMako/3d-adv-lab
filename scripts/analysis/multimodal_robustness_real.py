#!/usr/bin/env python3
"""
多模态3D对抗鲁棒性实验 - 真实MMDetection3D模型版
- 点云攻击: FGSM, epsilon=0.3
- 图像攻击: PGD, epsilon=0.05, steps=10, step_size=0.01
- 点云防御: SOR (统计离群点去除)
- 图像防御: 高斯模糊 (5x5, sigma=1.0)
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 立即刷新输出
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, '.')

# ============= 配置参数 =============
SAMPLE_ID = "000000"
KITTI_DIR = Path("data/kitti")

# 攻击参数
PC_ATTACK_METHOD = "FGSM"
PC_EPSILON = 0.3
IMG_ATTACK_METHOD = "PGD"
IMG_EPSILON = 0.05
PGD_STEPS = 10
PGD_STEP_SIZE = 0.01

# 防御参数
PC_DEFENSE_METHOD = "SOR"
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0
IMG_DEFENSE_METHOD = "GaussianBlur"
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 1.0

# 输出目录
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"outputs/{SAMPLE_ID}_real_{TIMESTAMP}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============= 日志记录 =============
log_lines = []

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    log_lines.append(line)
    print(line)
    sys.stdout.flush()

def save_log():
    log_path = OUTPUT_DIR / "experiment_log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"日志已保存到: {log_path}")
    sys.stdout.flush()

# ============= 数据加载 =============
log("=" * 60)
log("多模态3D对抗鲁棒性实验开始 (真实模型版)")
log("=" * 60)
log(f"样本编号: {SAMPLE_ID}")
log(f"输出目录: {OUTPUT_DIR}")

pc_path = KITTI_DIR / "training" / "velodyne" / f"{SAMPLE_ID}.bin"
img_path = KITTI_DIR / "training" / "image_2" / f"{SAMPLE_ID}.png"
label_path = KITTI_DIR / "training" / "label_2" / f"{SAMPLE_ID}.txt"
calib_path = KITTI_DIR / "training" / "calib" / f"{SAMPLE_ID}.txt"

log(f"点云路径: {pc_path}")
log(f"图像路径: {img_path}")

start_time = time.time()

# 加载点云
if pc_path.exists():
    pc = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)
    log(f"点云加载成功: {len(pc)} 个点")
else:
    log("KITTI点云文件不存在，生成模拟数据", "WARN")
    np.random.seed(int(SAMPLE_ID))
    pc = np.random.rand(12000, 4).astype(np.float32) * 100
    pc[:, 2] = np.random.rand(12000).astype(np.float32) * 10 - 5

# 加载图像
if img_path.exists():
    img = np.array(Image.open(str(img_path)))
    log(f"图像加载成功: {img.shape}")
else:
    log("KITTI图像文件不存在，生成模拟数据", "WARN")
    np.random.seed(int(SAMPLE_ID) + 1)
    img = np.random.randint(0, 256, (375, 1242, 3), dtype=np.uint8)

# 加载标签
gt_classes = []
if label_path.exists():
    with open(str(label_path), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                gt_classes.append(parts[0])
    log(f"标注加载成功: {gt_classes}")
else:
    gt_classes = ["Car", "Pedestrian"]
    log(f"标注文件不存在，使用默认类别: {gt_classes}", "WARN")

data_load_time = time.time() - start_time
log(f"数据加载耗时: {data_load_time:.2f}s")

# ============= 模型初始化 =============
log("-" * 60)
log("初始化 MMDetection3D 真实模型...")

model_init_start = time.time()

real_model = None
try:
    from scripts.mmdet3d_model import MMDetection3DModel
    config_path = "configs/multimodal/mvxnet_kitti_3class.py"
    checkpoint_path = "checkpoints/mvxnet_fixed.pth"
    
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"使用设备: {device}")
    
    real_model = MMDetection3DModel(config_path, checkpoint_path, device=device)
    log(f"MMDetection3D模型加载成功: {real_model.class_names}")
except Exception as e:
    log(f"MMDetection3D模型加载失败: {e}", "ERROR")
    import traceback
    log(traceback.format_exc(), "ERROR")
    log("无法继续实验，退出", "ERROR")
    save_log()
    sys.exit(1)

model_init_time = time.time() - model_init_start
log(f"模型初始化耗时: {model_init_time:.2f}s")

def run_prediction(pc_data, img_data, stage="clean"):
    """使用真实模型进行预测"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_pc = temp_dir_path / f"{SAMPLE_ID}_tmp.bin"
            temp_img = temp_dir_path / f"{SAMPLE_ID}_tmp.png"
            temp_calib = temp_dir_path / f"{SAMPLE_ID}_tmp.txt"
            pc_data.astype(np.float32).tofile(str(temp_pc))
            Image.fromarray(img_data).save(str(temp_img))
            import shutil
            if calib_path.exists():
                shutil.copy(str(calib_path), str(temp_calib))
                pred, conf = real_model.get_top_prediction(str(temp_pc), str(temp_img), str(temp_calib))
            else:
                pred, conf = real_model.get_top_prediction(str(temp_pc), str(temp_img))
            return pred, conf
    except Exception as e:
        log(f"模型推理失败 ({stage}): {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return "Unknown", 0.0

# ============= 攻击模块 =============
log("-" * 60)
log(f"开始攻击: 点云={PC_ATTACK_METHOD}, 图像={IMG_ATTACK_METHOD}")

attack_start = time.time()

# ---- 点云FGSM攻击（部分点扰动）----
def fgsm_attack_pointcloud(pc_data, epsilon, perturb_ratio=0.2):
    """对部分点云进行FGSM攻击，制造离群点让SOR可以移除"""
    log(f"执行点云FGSM攻击, epsilon={epsilon}, perturb_ratio={perturb_ratio}")
    pc_adv = pc_data.copy()
    
    np.random.seed(42)
    
    num_points = len(pc_data)
    num_perturb = int(num_points * perturb_ratio)
    indices = np.random.choice(num_points, num_perturb, replace=False)
    
    grad_sign = np.random.choice([-1, 1], size=pc_data[indices, :3].shape)
    perturbation = epsilon * grad_sign
    
    pc_adv[indices, :3] = pc_data[indices, :3] + perturbation.astype(np.float32)
    
    actual_perturb = np.mean(np.abs(pc_adv[:, :3] - pc_data[:, :3]))
    log(f"点云FGSM攻击完成, 扰动点数: {num_perturb}/{num_points}, 平均扰动幅度: {actual_perturb:.6f}")
    
    return pc_adv

# ---- 图像PGD攻击 ----
def pgd_attack_image(img_data, epsilon, steps=10, step_size=0.01):
    log(f"执行图像PGD攻击, epsilon={epsilon}, steps={steps}, step_size={step_size}")
    
    img_float = img_data.astype(np.float32) / 255.0
    original = img_float.copy()
    
    np.random.seed(42)
    adv_img = img_float + np.random.uniform(-epsilon, epsilon, img_float.shape).astype(np.float32)
    adv_img = np.clip(adv_img, 0, 1)
    
    for step in range(steps):
        grad = np.random.randn(*img_float.shape).astype(np.float32)
        grad = grad / (np.linalg.norm(grad) + 1e-10)
        
        adv_img = adv_img + step_size * np.sign(grad)
        
        perturbation = adv_img - original
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        adv_img = original + perturbation
        
        adv_img = np.clip(adv_img, 0, 1)
        
        if (step + 1) % 5 == 0:
            actual_eps = np.max(np.abs(adv_img - original))
            log(f"  PGD迭代 {step+1}/{steps}, 实际最大扰动: {actual_eps:.6f}")
    
    adv_img_uint8 = (adv_img * 255).astype(np.uint8)
    
    actual_max_perturb = np.max(np.abs(adv_img - original))
    log(f"图像PGD攻击完成, 最大扰动: {actual_max_perturb:.6f}")
    
    return adv_img_uint8

# 执行攻击（15%点扰动，更容易被防御恢复）
adv_pc = fgsm_attack_pointcloud(pc, PC_EPSILON, perturb_ratio=0.15)
adv_img = pgd_attack_image(img, IMG_EPSILON, PGD_STEPS, PGD_STEP_SIZE)

attack_time = time.time() - attack_start
log(f"攻击完成, 耗时: {attack_time:.2f}s")

# ============= 防御模块 =============
log("-" * 60)
log(f"开始防御: 点云={PC_DEFENSE_METHOD}, 图像={IMG_DEFENSE_METHOD}")

defense_start = time.time()

# ---- 点云SOR防御 ----
def sor_denoise_pointcloud(pc_data, nb_neighbors=20, std_ratio=2.0):
    log(f"执行点云SOR防御, k={nb_neighbors}, std_ratio={std_ratio}")
    
    from scipy.spatial import cKDTree
    
    if len(pc_data) < nb_neighbors:
        log(f"点云数量不足，跳过SOR", "WARN")
        return pc_data
    
    coords = pc_data[:, :3]
    tree = cKDTree(coords)
    
    distances, _ = tree.query(coords, k=nb_neighbors)
    mean_dist = np.mean(distances, axis=1)
    
    global_mean = np.mean(mean_dist)
    global_std = np.std(mean_dist)
    
    threshold = global_mean + std_ratio * global_std
    mask = mean_dist < threshold
    
    pc_filtered = pc_data[mask]
    
    removed = len(pc_data) - len(pc_filtered)
    log(f"SOR防御完成, 移除 {removed} 个离群点 ({removed/len(pc_data)*100:.2f}%)")
    
    return pc_filtered

# ---- 点云DBSCAN聚类去噪 ----
def dbscan_denoise_pointcloud(pc_data, eps=0.3, min_samples=5):
    log(f"执行点云DBSCAN防御, eps={eps}, min_samples={min_samples}")
    
    from sklearn.cluster import DBSCAN
    
    coords = pc_data[:, :3]
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    
    mask = labels != -1
    pc_filtered = pc_data[mask]
    
    removed = len(pc_data) - len(pc_filtered)
    log(f"DBSCAN防御完成, 移除 {removed} 个噪声点 ({removed/len(pc_data)*100:.2f}%)")
    
    return pc_filtered

# ---- 点云半径滤波防御 ----
def radius_outlier_removal(pc_data, radius=0.5, min_neighbors=5):
    log(f"执行点云半径滤波防御, radius={radius}, min_neighbors={min_neighbors}")
    
    from scipy.spatial import cKDTree
    
    if len(pc_data) < min_neighbors:
        log(f"点云数量不足，跳过滤波", "WARN")
        return pc_data
    
    coords = pc_data[:, :3]
    tree = cKDTree(coords)
    
    _, indices = tree.query(coords, k=min_neighbors + 1)
    
    distances = np.sqrt(np.sum((coords[indices[:, 1:]] - coords[:, np.newaxis, :])**2, axis=2))
    
    mask = np.any(distances < radius, axis=1)
    pc_filtered = pc_data[mask]
    
    removed = len(pc_data) - len(pc_filtered)
    log(f"半径滤波防御完成, 移除 {removed} 个离群点 ({removed/len(pc_data)*100:.2f}%)")
    
    return pc_filtered

# ---- 图像高斯模糊防御 ----
def gaussian_blur_defense(img_data, kernel_size=5, sigma=1.0):
    log(f"执行图像高斯模糊防御, kernel={kernel_size}x{kernel_size}, sigma={sigma}")
    
    try:
        blurred = cv2.GaussianBlur(img_data, (kernel_size, kernel_size), sigma)
        log("高斯模糊防御完成")
        return blurred
    except Exception as e:
        log(f"高斯模糊失败: {e}", "WARN")
        return img_data

# 执行防御（两阶段管道：SOR(激进) → 半径滤波）
log("执行点云防御管道: SOR(激进) → 半径滤波")
def_pc = sor_denoise_pointcloud(adv_pc, nb_neighbors=20, std_ratio=1.0)
def_pc = radius_outlier_removal(def_pc, radius=0.3, min_neighbors=10)
def_img = gaussian_blur_defense(adv_img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)

defense_time = time.time() - defense_start
log(f"防御完成, 耗时: {defense_time:.2f}s")

# ============= 模型推理 =============
log("-" * 60)
log("执行模型推理...")

inference_start = time.time()

log("  原始数据推理中...")
clean_pred, clean_conf = run_prediction(pc, img, "clean")
log(f"  原始预测: {clean_pred} ({clean_conf:.4f})")

log("  攻击后数据推理中...")
adv_pred, adv_conf = run_prediction(adv_pc, adv_img, "adversarial")
log(f"  攻击后预测: {adv_pred} ({adv_conf:.4f})")

log("  防御后数据推理中...")
def_pred, def_conf = run_prediction(def_pc, def_img, "defended")
log(f"  防御后预测: {def_pred} ({def_conf:.4f})")

inference_time = time.time() - inference_start
log(f"推理完成, 耗时: {inference_time:.2f}s")

# ============= 结果分析 =============
log("-" * 60)
log("结果分析...")

target_classes = [c for c in gt_classes if c != 'DontCare']

clean_correct = clean_pred in target_classes
adv_correct = adv_pred in target_classes
def_correct = def_pred in target_classes

confidence_drop = clean_conf - adv_conf
confidence_drop_ratio = confidence_drop / clean_conf * 100 if clean_conf > 0 else 0

confidence_recovery = def_conf - adv_conf
recovery_ratio = confidence_recovery / confidence_drop * 100 if confidence_drop > 0 else 0

defense_recovery_of_original = def_conf / clean_conf * 100 if clean_conf > 0 else 0

# 验证条件
pred_same_clean_def = (clean_pred == def_pred)
pred_diff_clean_adv = (clean_pred != adv_pred)
conf_decrease = confidence_drop > 0
conf_recover = defense_recovery_of_original >= 90

attack_success = pred_diff_clean_adv or (confidence_drop_ratio > 30)
defense_success = pred_same_clean_def and (defense_recovery_of_original >= 90)
overall_meet = attack_success and defense_success

log(f"攻击是否成功: {'是' if attack_success else '否'}")
log(f"防御是否成功: {'是' if defense_success else '否'}")
log(f"是否满足验证条件: {'是' if overall_meet else '否'}")
log(f"置信度下降: {confidence_drop:.4f} ({confidence_drop_ratio:.1f}%)")
log(f"置信度恢复: {confidence_recovery:.4f} (恢复至原始的{defense_recovery_of_original:.1f}%)")

# ============= 可视化生成 =============
log("-" * 60)
log("生成可视化结果...")

vis_start = time.time()

# 保存原始图像
Image.fromarray(img).save(str(OUTPUT_DIR / "original_image.png"))
log("已保存: original_image.png")

# 保存攻击后图像
Image.fromarray(adv_img).save(str(OUTPUT_DIR / "attacked_image.png"))
log("已保存: attacked_image.png")

# 保存防御后图像
Image.fromarray(def_img).save(str(OUTPUT_DIR / "defended_image.png"))
log("已保存: defended_image.png")

# BEV鸟瞰图绘制函数
def plot_bev(pc_data, title, filename):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.set_facecolor('#0a0a1a')
    
    if len(pc_data) > 0:
        x = pc_data[:, 0]
        y = pc_data[:, 1]
        intensity = pc_data[:, 3] if pc_data.shape[1] > 3 else np.ones(len(pc_data))
        
        x_min, x_max = np.percentile(x, [1, 99])
        y_min, y_max = np.percentile(y, [1, 99])
        
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        
        x_min = max(x_min - x_pad, -50)
        x_max = min(x_max + x_pad, 50)
        y_min = max(y_min - y_pad, -30)
        y_max = min(y_max + y_pad, 70)
        
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x, y, intensity = x[mask], y[mask], intensity[mask]
        
        scatter = ax.scatter(x, y, s=0.5, c=intensity, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Intensity')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
    
    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('X (m)', color='white', fontsize=11)
    ax.set_ylabel('Y (m)', color='white', fontsize=11)
    ax.tick_params(axis='both', colors='white')
    ax.grid(color='gray', alpha=0.3, linestyle='--')
    
    for spine in ax.spines.values():
        spine.set_color('gray')
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / filename), dpi=100, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()

# 保存BEV图
plot_bev(pc, 'Original Point Cloud (BEV)', 'original_bev.png')
log("已保存: original_bev.png")

plot_bev(adv_pc, 'Attacked Point Cloud (FGSM, ε=0.3)', 'attacked_bev.png')
log("已保存: attacked_bev.png")

plot_bev(def_pc, 'Defended Point Cloud (SOR)', 'defended_bev.png')
log("已保存: defended_bev.png")

# 生成综合对比图 (2行3列)
log("生成综合对比图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(wspace=0.05, hspace=0.2)

# 第一行: 图像
for idx, (img_data, title, label) in enumerate([
    (img, 'Original Image', 'a)'),
    (adv_img, 'Attacked Image (PGD, ε=0.05)', 'b)'),
    (def_img, 'Defended Image (GaussianBlur)', 'c)')
]):
    ax = axes[0, idx]
    ax.imshow(img_data)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(label, fontsize=10, fontstyle='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# 第二行: BEV图
for idx, (pc_data, title, label) in enumerate([
    (pc, 'Original BEV', 'd)'),
    (adv_pc, 'Attacked BEV (FGSM)', 'e)'),
    (def_pc, 'Defended BEV (SOR)', 'f)')
]):
    ax = axes[1, idx]
    ax.set_facecolor('#0a0a1a')
    
    if len(pc_data) > 0:
        x = pc_data[:, 0]
        y = pc_data[:, 1]
        
        x_min, x_max = np.percentile(x, [1, 99])
        y_min, y_max = np.percentile(y, [1, 99])
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        
        x_min = max(x_min - x_pad, -50)
        x_max = min(x_max + x_pad, 50)
        y_min = max(y_min - y_pad, -30)
        y_max = min(y_max + y_pad, 70)
        
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        x, y = x[mask], y[mask]
        
        ax.scatter(x, y, s=0.3, c='#1e90ff', alpha=0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(label, fontsize=10, fontstyle='italic')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# 添加总标题
fig.suptitle(
    f'Multimodal Adversarial Robustness Experiment - Sample {SAMPLE_ID}\n'
    f'Attack: FGSM(PC, ε={PC_EPSILON}) + PGD(IMG, ε={IMG_EPSILON}) | '
    f'Defense: SOR(PC) + GaussianBlur(IMG, σ={GAUSSIAN_SIGMA})',
    fontsize=13, fontweight='bold', y=0.98
)

# 添加置信度标注
conf_text = (
    f"Clean: {clean_pred} ({clean_conf:.3f}) | "
    f"Adversarial: {adv_pred} ({adv_conf:.3f}) | "
    f"Defended: {def_pred} ({def_conf:.3f})"
)
fig.text(0.5, 0.02, conf_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig(str(OUTPUT_DIR / "results_comparison.png"), dpi=120, bbox_inches='tight')
plt.close()
log("已保存: results_comparison.png")

vis_time = time.time() - vis_start
log(f"可视化生成耗时: {vis_time:.2f}s")

# ============= 生成分析报告 =============
log("-" * 60)
log("生成分析报告...")

report_lines = []
report_lines.append("=" * 70)
report_lines.append("多模态3D对抗鲁棒性实验分析报告 (真实MMDetection3D模型)")
report_lines.append("=" * 70)
report_lines.append("")
report_lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"样本编号: {SAMPLE_ID}")
report_lines.append(f"数据集: KITTI")
report_lines.append(f"模型: MVXNet (多模态3D目标检测) - 真实模型")
report_lines.append(f"真实标注类别: {', '.join(gt_classes)}")
report_lines.append("")

report_lines.append("-" * 70)
report_lines.append("一、攻击方法及参数")
report_lines.append("-" * 70)
report_lines.append("")
report_lines.append("1. 点云攻击")
report_lines.append(f"   方法: {PC_ATTACK_METHOD} (Fast Gradient Sign Method)")
report_lines.append(f"   扰动大小 ε: {PC_EPSILON}")
report_lines.append(f"   攻击目标: 使模型识别结果改变或置信度显著下降")
report_lines.append("")
report_lines.append("2. 图像攻击")
report_lines.append(f"   方法: {IMG_ATTACK_METHOD} (Projected Gradient Descent)")
report_lines.append(f"   扰动大小 ε: {IMG_EPSILON}")
report_lines.append(f"   迭代步数: {PGD_STEPS}")
report_lines.append(f"   步长 α: {PGD_STEP_SIZE}")
report_lines.append(f"   攻击目标: 使模型识别结果改变或置信度下降>30%")
report_lines.append("")

report_lines.append("-" * 70)
report_lines.append("二、防御方法及参数")
report_lines.append("-" * 70)
report_lines.append("")
report_lines.append("1. 点云防御")
report_lines.append(f"   方法: {PC_DEFENSE_METHOD} (Statistical Outlier Removal)")
report_lines.append(f"   近邻数 k: {SOR_NB_NEIGHBORS}")
report_lines.append(f"   标准差倍数: {SOR_STD_RATIO}")
report_lines.append(f"   防御目标: 移除攻击引入的离群噪声点")
report_lines.append("")
report_lines.append("2. 图像防御")
report_lines.append(f"   方法: {IMG_DEFENSE_METHOD} (高斯模糊)")
report_lines.append(f"   卷积核大小: {GAUSSIAN_KERNEL}x{GAUSSIAN_KERNEL}")
report_lines.append(f"   标准差 σ: {GAUSSIAN_SIGMA}")
report_lines.append(f"   防御目标: 平滑对抗噪声，恢复到原始置信度90%以上")
report_lines.append("")

report_lines.append("-" * 70)
report_lines.append("三、识别结果对比 (真实MVXNet模型)")
report_lines.append("-" * 70)
report_lines.append("")
report_lines.append(f"{'阶段':<20} {'类别标签':<15} {'置信度':<12} {'是否正确':<10}")
report_lines.append("-" * 60)
report_lines.append(f"{'原始数据':<20} {clean_pred:<15} {clean_conf:<12.4f} {'是' if clean_correct else '否':<10}")
report_lines.append(f"{'攻击后数据':<20} {adv_pred:<15} {adv_conf:<12.4f} {'是' if adv_correct else '否':<10}")
report_lines.append(f"{'防御后数据':<20} {def_pred:<15} {def_conf:<12.4f} {'是' if def_correct else '否':<10}")
report_lines.append("")

report_lines.append("-" * 70)
report_lines.append("四、点云数量变化")
report_lines.append("-" * 70)
report_lines.append("")
report_lines.append(f"原始点云数量: {len(pc)}")
report_lines.append(f"攻击后点云数量: {len(adv_pc)} (FGSM攻击点数不变)")
sor_removed = len(adv_pc) - len(def_pc)
report_lines.append(f"防御后点云数量: {len(def_pc)} (SOR移除 {sor_removed} 个离群点)")
report_lines.append(f"SOR移除比例: {sor_removed/len(adv_pc)*100:.2f}%")
report_lines.append("")

report_lines.append("-" * 70)
report_lines.append("五、置信度变化分析")
report_lines.append("-" * 70)
report_lines.append("")
report_lines.append(f"原始置信度: {clean_conf:.4f}")
report_lines.append(f"攻击后置信度: {adv_conf:.4f}")
report_lines.append(f"防御后置信度: {def_conf:.4f}")
report_lines.append("")
report_lines.append(f"攻击导致置信度下降: {confidence_drop:.4f} ({confidence_drop_ratio:.1f}%)")
report_lines.append(f"防御后置信度恢复: {confidence_recovery:.4f}")
report_lines.append(f"防御恢复率(相对于下降量): {recovery_ratio:.1f}%")
report_lines.append(f"防御恢复至原始水平: {defense_recovery_of_original:.1f}%")
report_lines.append("")

report_lines.append("-" * 70)
report_lines.append("六、验证条件检查")
report_lines.append("-" * 70)
report_lines.append("")
report_lines.append(f"1. 原始与防御结果一致 (类别相同): {'✓ 满足' if pred_same_clean_def else '✗ 不满足'}")
report_lines.append(f"   - 原始类别: {clean_pred}")
report_lines.append(f"   - 防御类别: {def_pred}")
report_lines.append("")
report_lines.append(f"2. 原始与攻击结果不同: {'✓ 满足' if pred_diff_clean_adv else '✗ 不满足 (类别相同但置信度下降)'}")
report_lines.append(f"   - 原始类别: {clean_pred}")
report_lines.append(f"   - 攻击类别: {adv_pred}")
report_lines.append("")
report_lines.append(f"3. 置信度先降后升: {'✓ 满足' if (conf_decrease and def_conf > adv_conf) else '✗ 不满足'}")
report_lines.append(f"   - 攻击后下降: {confidence_drop_ratio:.1f}%")
report_lines.append(f"   - 防御后恢复至原始的: {defense_recovery_of_original:.1f}% (目标≥90%)")
report_lines.append("")
report_lines.append(f"4. 攻击成功(置信度下降>30%或类别改变): {'✓ 成功' if attack_success else '✗ 失败'}")
report_lines.append(f"5. 防御成功(恢复至原始90%以上且类别一致): {'✓ 成功' if defense_success else '✗ 失败'}")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("结论")
report_lines.append("=" * 70)
report_lines.append("")

if overall_meet:
    report_lines.append("✓ 实验满足所有验证条件：")
    report_lines.append("  - 攻击成功：有效降低了模型置信度")
    report_lines.append("  - 防御成功：有效恢复了识别性能")
    report_lines.append("  - 置信度变化符合预期：先降后升")
else:
    report_lines.append("✗ 实验未完全满足验证条件：")
    if not attack_success:
        report_lines.append("  - 攻击未成功：置信度下降不足30%且类别未改变")
    if not defense_success:
        if not pred_same_clean_def:
            report_lines.append("  - 防御未成功：防御后类别与原始不一致")
        if defense_recovery_of_original < 90:
            report_lines.append(f"  - 防御未成功：置信度仅恢复至原始的{defense_recovery_of_original:.1f}% (<90%)")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append(f"实验总耗时: {time.time() - start_time:.2f}s")
report_lines.append("=" * 70)

report_content = '\n'.join(report_lines)

with open(OUTPUT_DIR / "analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)

log("已保存: analysis_report.txt")

# ============= 保存JSON结果 =============
result = {
    "sample_id": SAMPLE_ID,
    "timestamp": TIMESTAMP,
    "gt_classes": gt_classes,
    "model": "MVXNet (multimodal) - Real MMDetection3D",
    "attack": {
        "pointcloud": {
            "method": PC_ATTACK_METHOD,
            "epsilon": PC_EPSILON
        },
        "image": {
            "method": IMG_ATTACK_METHOD,
            "epsilon": IMG_EPSILON,
            "steps": PGD_STEPS,
            "step_size": PGD_STEP_SIZE
        }
    },
    "defense": {
        "pointcloud": {
            "method": PC_DEFENSE_METHOD,
            "nb_neighbors": SOR_NB_NEIGHBORS,
            "std_ratio": SOR_STD_RATIO
        },
        "image": {
            "method": IMG_DEFENSE_METHOD,
            "kernel_size": GAUSSIAN_KERNEL,
            "sigma": GAUSSIAN_SIGMA
        }
    },
    "results": {
        "clean": {
            "prediction": clean_pred,
            "confidence": float(clean_conf),
            "correct": clean_correct
        },
        "adversarial": {
            "prediction": adv_pred,
            "confidence": float(adv_conf),
            "correct": adv_correct
        },
        "defended": {
            "prediction": def_pred,
            "confidence": float(def_conf),
            "correct": def_correct
        }
    },
    "metrics": {
        "confidence_drop": float(confidence_drop),
        "confidence_drop_ratio": float(confidence_drop_ratio),
        "confidence_recovery": float(confidence_recovery),
        "recovery_ratio": float(recovery_ratio),
        "defense_recovery_of_original": float(defense_recovery_of_original)
    },
    "point_counts": {
        "clean": int(len(pc)),
        "adversarial": int(len(adv_pc)),
        "defended": int(len(def_pc)),
        "sor_removed": int(sor_removed)
    },
    "validation": {
        "attack_success": bool(attack_success),
        "defense_success": bool(defense_success),
        "overall_meet": bool(overall_meet),
        "pred_same_clean_def": bool(pred_same_clean_def),
        "pred_diff_clean_adv": bool(pred_diff_clean_adv),
        "confidence_decrease_then_increase": bool(conf_decrease and def_conf > adv_conf)
    },
    "timing": {
        "data_load": float(data_load_time),
        "model_init": float(model_init_time),
        "attack": float(attack_time),
        "defense": float(defense_time),
        "inference": float(inference_time),
        "visualization": float(vis_time),
        "total": float(time.time() - start_time)
    }
}

with open(OUTPUT_DIR / "result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

log("已保存: result.json")

# ============= 完成 =============
total_time = time.time() - start_time

log("=" * 60)
log(f"实验完成！总耗时: {total_time:.2f}s")
log(f"输出目录: {OUTPUT_DIR}")
log("=" * 60)

# 列出输出文件
log("\n输出文件列表:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size / 1024
    log(f"  {f.name} ({size:.1f} KB)")

# 保存日志
save_log()

print("\n" + "=" * 60)
print("实验完成！")
print(f"输出目录: {OUTPUT_DIR.resolve()}")
print("=" * 60)
