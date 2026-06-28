#!/usr/bin/env python3
"""
Prepare KITTI dataset info files for MMDetection3D.
Generates kitti_infos_train.pkl and kitti_infos_val.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare KITTI infos")
    parser.add_argument("--data-root", default="data/kitti", help="KITTI root")
    parser.add_argument("--train-split", default="ImageSets/train.txt", help="Train split file")
    parser.add_argument("--val-split", default="ImageSets/val.txt", help="Val split file")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    
    train_ids = []
    with open(root / args.train_split, "r", encoding='utf-8-sig') as f:
        for line in f:
            train_ids.append(line.strip())
    
    val_ids = []
    with open(root / args.val_split, "r", encoding='utf-8-sig') as f:
        for line in f:
            val_ids.append(line.strip())
    
    def generate_infos(sample_ids, split):
        infos = []
        for idx, sample_id in enumerate(sample_ids):
            info = {
                "image": {
                    "image_idx": int(sample_id),
                    "image_path": f"training/image_2/{sample_id}.png",
                    "image_shape": [375, 1242, 3],
                },
                "point_cloud": {
                    "num_features": 4,
                    "velodyne_path": f"training/velodyne/{sample_id}.bin",
                },
                "calib": {
                    "P0": np.eye(4).flatten().tolist(),
                    "P1": np.eye(4).flatten().tolist(),
                    "P2": np.eye(4).flatten().tolist(),
                    "P3": np.eye(4).flatten().tolist(),
                    "R0_rect": np.eye(4).flatten().tolist(),
                    "Tr_velo_to_cam": np.eye(4).flatten().tolist(),
                    "Tr_imu_to_velo": np.eye(4).flatten().tolist(),
                },
                "annos": {
                    "name": [],
                    "truncated": [],
                    "occluded": [],
                    "alpha": [],
                    "bbox": [],
                    "dimensions": [],
                    "location": [],
                    "rotation_y": [],
                    "score": [],
                    "index": [],
                    "group_ids": [],
                    "difficulty": [],
                    "num_points_in_gt": [],
                },
                "timestamp": idx,
                "idx": idx,
            }
            
            label_path = root / "training" / "label_2" / f"{sample_id}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        name = parts[0]
                        truncated = float(parts[1])
                        occluded = int(parts[2])
                        alpha = float(parts[3])
                        bbox = list(map(float, parts[4:8]))
                        dimensions = list(map(float, parts[8:11]))
                        location = list(map(float, parts[11:14]))
                        rotation_y = float(parts[14])
                        
                        info["annos"]["name"].append(name)
                        info["annos"]["truncated"].append(truncated)
                        info["annos"]["occluded"].append(occluded)
                        info["annos"]["alpha"].append(alpha)
                        info["annos"]["bbox"].append(bbox)
                        info["annos"]["dimensions"].append(dimensions)
                        info["annos"]["location"].append(location)
                        info["annos"]["rotation_y"].append(rotation_y)
                        info["annos"]["difficulty"].append(0)
            
            infos.append(info)
        
        output_path = root / f"kitti_infos_{split}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(infos, f)
        
        print(f"Generated {len(infos)} infos to {output_path}")
        return infos
    
    print("Generating training infos...")
    generate_infos(train_ids, "train")
    
    print("Generating validation infos...")
    generate_infos(val_ids, "val")


if __name__ == "__main__":
    main()