#!/usr/bin/env python3
"""Test MMDetection3D multi-modal model inference."""

import numpy as np
from mmdet3d.apis import init_model, inference_multi_modality_detector

# Load model
model = init_model('configs/multimodal/mvxnet_kitti_3class.py', device='cpu')
model.eval()

# Test with both point cloud and image
pc_path = 'data/kitti/training/velodyne/000000.bin'
img_path = 'data/kitti/training/image_2/000000.png'
ann_file = 'data/kitti/kitti_infos_val.pkl'

try:
    result, data = inference_multi_modality_detector(model, pc_path, img_path, ann_file, cam_type='CAM2')
    print(f"Result type: {type(result)}")
    
    # Check the structure
    if hasattr(result, 'pred_instances_3d'):
        pred_instances = result.pred_instances_3d
        print(f"Pred instances type: {type(pred_instances)}")
        if hasattr(pred_instances, 'labels'):
            print(f"Labels: {pred_instances.labels}")
        if hasattr(pred_instances, 'scores'):
            print(f"Scores: {pred_instances.scores}")
    else:
        print(f"Result content keys: {result.keys() if hasattr(result, 'keys') else 'No keys'}")
        
except Exception as e:
    print(f"Inference failed: {e}")
    import traceback
    traceback.print_exc()