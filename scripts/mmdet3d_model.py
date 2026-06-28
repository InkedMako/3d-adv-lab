#!/usr/bin/env python3
"""
Real MMDetection3D Model Inference Module
Uses MVXNet for multi-modal 3D detection on KITTI dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.structures import get_box_type
    from mmcv.transforms import Compose
    from mmengine.dataset import pseudo_collate
    from copy import deepcopy
    import numpy as np
    HAS_MMDET3D = True
except ImportError:
    HAS_MMDET3D = False


class MMDetection3DModel:
    """Wrapper for MMDetection3D multi-modal models."""
    
    def __init__(self, config_path: str, checkpoint_path: str = None, device: str = "cuda"):
        if not HAS_MMDET3D:
            raise ImportError("MMDetection3D is not installed. Please install it first.")
        
        self.config_path = config_path
        self.device = device
        self.model = self._init_model(checkpoint_path)
        self.class_names = self._get_class_names()
    
    def _init_model(self, checkpoint_path: str = None):
        """Initialize the model from config and checkpoint."""
        if checkpoint_path is None:
            # Use pretrained checkpoint from config
            model = init_model(self.config_path, device=self.device)
        else:
            model = init_model(self.config_path, checkpoint_path, device=self.device)
        model.eval()
        return model
    
    def _get_class_names(self):
        """Get class names from model config."""
        if hasattr(self.model, 'module'):
            cfg = self.model.module.cfg
        else:
            cfg = self.model.cfg
        
        if hasattr(cfg, 'metainfo') and cfg.metainfo is not None:
            return cfg.metainfo.get('classes', ["Car", "Pedestrian", "Cyclist"])
        return ["Car", "Pedestrian", "Cyclist"]
    
    @torch.no_grad()
    def inference(self, pointcloud_path: str, image_path: str = None, calib_path: str = None) -> dict:
        """
        Run inference on a single sample.
        
        Args:
            pointcloud_path: Path to .bin point cloud file
            image_path: Path to .png image file (optional, for multi-modal)
            calib_path: Path to calibration file (optional, for KITTI multi-modal)
        
        Returns:
            Dictionary with predictions (class names, scores, bboxes)
        """
        cfg = self.model.cfg
        
        if image_path is not None:
            test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
            test_pipeline = Compose(test_pipeline)
            box_type_3d, box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)
            
            data_ = dict(
                lidar_points=dict(lidar_path=pointcloud_path),
                img_path=image_path,
                timestamp=1,
                axis_align_matrix=np.eye(4),
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
            
            if calib_path is not None:
                calib = self._load_kitti_calib(calib_path)
                data_.update(calib)
            
            data_ = test_pipeline(data_)
            collate_data = pseudo_collate([data_])
            
            with torch.no_grad():
                results = self.model.test_step(collate_data)
            
            result = results[0]
        else:
            result, _ = inference_detector(self.model, pointcloud_path)
        
        pred_instances = result.pred_instances_3d
        
        if pred_instances is None:
            return {
                'predictions': [],
                'scores': [],
                'bboxes_3d': [],
                'has_detection': False
            }
        
        def _get_tensor(name):
            if hasattr(pred_instances, name):
                val = getattr(pred_instances, name)
                if val is not None:
                    return val.cpu().numpy()
            return np.array([])
        
        labels = _get_tensor('labels_3d') if len(_get_tensor('labels_3d')) > 0 else _get_tensor('labels')
        scores = _get_tensor('scores_3d') if len(_get_tensor('scores_3d')) > 0 else _get_tensor('scores')
        bboxes_3d = _get_tensor('bboxes_3d')
        
        if len(scores) == 0 and len(labels) == 0:
            return {
                'predictions': [],
                'scores': [],
                'bboxes_3d': [],
                'has_detection': False
            }
        
        predictions = []
        for label in labels:
            if label < len(self.class_names):
                predictions.append(self.class_names[label])
            else:
                predictions.append(f"Unknown_{label}")
        
        return {
            'predictions': predictions,
            'scores': scores.tolist() if len(scores) > 0 else [],
            'bboxes_3d': bboxes_3d.tolist() if len(bboxes_3d) > 0 else [],
            'has_detection': len(predictions) > 0,
            'class_names': self.class_names
        }
    
    def _load_kitti_calib(self, calib_path: str) -> dict:
        """Load KITTI calibration file and compute projection matrices."""
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        
        calib = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
        
        P2 = calib['P2'].reshape(3, 4)
        R0_rect = np.eye(4)
        R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[:3, :] = calib['Tr_velo_to_cam'].reshape(3, 4)
        
        lidar2img = P2 @ R0_rect @ Tr_velo_to_cam
        
        return {
            'lidar2img': lidar2img,
            'cam2img': P2[:3, :3],
            'lidar2cam': R0_rect @ Tr_velo_to_cam,
        }
    
    def get_top_prediction(self, pointcloud_path: str, image_path: str = None, calib_path: str = None) -> tuple[str, float]:
        """
        Get the top prediction (highest confidence).
        
        Returns:
            (class_name, confidence)
        """
        result = self.inference(pointcloud_path, image_path, calib_path)
        
        if not result['has_detection'] or len(result['scores']) == 0:
            return "Unknown", 0.0
        
        max_idx = np.argmax(result['scores'])
        return result['predictions'][max_idx], result['scores'][max_idx]


def test_model():
    """Test the model inference."""
    parser = argparse.ArgumentParser(description="Test MMDetection3D model")
    parser.add_argument("--config", default="configs/multimodal/mvxnet_kitti_3class.py", help="Model config")
    parser.add_argument("--pc-path", required=True, help="Point cloud path")
    parser.add_argument("--img-path", required=True, help="Image path")
    args = parser.parse_args()
    
    print(f"Initializing model from {args.config}...")
    model = MMDetection3DModel(args.config)
    print(f"Class names: {model.class_names}")
    
    print(f"\nRunning inference on {args.pc_path} and {args.img_path}...")
    result = model.inference(args.pc_path, args.img_path)
    
    print("\nInference result:")
    print(f"Has detection: {result['has_detection']}")
    if result['has_detection']:
        for i, (pred, score) in enumerate(zip(result['predictions'], result['scores'])):
            print(f"  [{i}] {pred}: {score:.4f}")
        
        top_pred, top_score = model.get_top_prediction(args.pc_path, args.img_path)
        print(f"\nTop prediction: {top_pred} (confidence: {top_score:.4f})")


if __name__ == "__main__":
    test_model()