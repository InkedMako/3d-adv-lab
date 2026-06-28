#!/usr/bin/env python3
"""
检查项目状态：模型加载、推理等
"""
import os
import sys

def check_environment():
    """检查环境"""
    print("=== 环境检查 ===")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查CUDA
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"PyTorch检查失败: {e}")
    
    # 检查mmdet3d
    try:
        import mmdet3d
        print(f"MMDetection3D版本: {mmdet3d.__version__}")
    except Exception as e:
        print(f"MMDetection3D检查失败: {e}")

def check_files():
    """检查关键文件"""
    print("\n=== 文件检查 ===")
    
    files_to_check = [
        'checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py',
        'checkpoints/mvxnet_fixed.pth',
        'data/kitti/kitti_infos_train.pkl',
    ]
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✓ {filepath} ({size:.2f} MB)")
        else:
            print(f"✗ {filepath} 不存在")

def check_model_load():
    """检查模型加载"""
    print("\n=== 模型加载检查 ===")
    
    from mmdet3d.apis import init_model
    
    config_file = 'checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py'
    checkpoint_file = 'checkpoints/mvxnet_fixed.pth'
    
    try:
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        print(f"✓ MVXNet模型加载成功")
        print(f"  模型类型: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

def check_inference():
    """检查推理功能"""
    print("\n=== 推理功能检查 ===")
    
    from mmdet3d.apis import init_model, inference_multi_modality_detector
    import numpy as np
    import cv2
    
    config_file = 'checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py'
    checkpoint_file = 'checkpoints/mvxnet_fixed.pth'
    ann_file = 'data/kitti/kitti_infos_train.pkl'
    
    try:
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        
        # 使用第一个样本测试
        sample_id = '000000'
        pc_path = f'data/kitti/training/velodyne/{sample_id}.bin'
        img_path = f'data/kitti/training/image_2/{sample_id}.png'
        
        if not os.path.exists(pc_path):
            print(f"✗ 点云文件不存在: {pc_path}")
            return
        
        if not os.path.exists(img_path):
            print(f"✗ 图像文件不存在: {img_path}")
            return
        
        # 加载数据
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(img_path)
        
        print(f"点云点数: {len(pc)}")
        print(f"图像尺寸: {img.shape}")
        
        # 执行推理
        result = inference_multi_modality_detector(model, pc_path, img_path, ann_file)
        
        if result is not None and isinstance(result, tuple) and len(result) > 0:
            result = result[0]
            if hasattr(result, 'pred_instances_3d'):
                pred_instances = result.pred_instances_3d
                num_detections = len(pred_instances.scores_3d)
                print(f"✓ 推理成功，检测到 {num_detections} 个目标")
            else:
                print(f"✗ 推理结果格式异常")
        else:
            print(f"✗ 推理返回None")
            
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_environment()
    check_files()
    check_model_load()
    check_inference()
    
    print("\n=== 检查完成 ===")