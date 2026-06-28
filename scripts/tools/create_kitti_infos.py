#!/usr/bin/env python3
"""
生成KITTI数据集标注信息文件
用于MVXNet多模态推理
"""
import os
import pickle
import numpy as np
from pathlib import Path

def parse_calib_file(calib_path):
    """解析calib文件"""
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    calib = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        if len(parts) != 2:
            continue
        key = parts[0]
        values = parts[1].strip().split()
        calib[key] = np.array([float(v) for v in values])

    return calib

def compute_lidar2img(cam2img, R0_rect, Tr_velo_to_cam):
    """计算lidar到图像的投影矩阵"""
    cam2img = np.array(cam2img).reshape(3, 4)
    R0_rect = np.array(R0_rect).reshape(3, 3)
    Tr_velo_to_cam = np.array(Tr_velo_to_cam).reshape(3, 4)

    # 计算 R0_rect @ Tr_velo_to_cam (3x4)
    R0_tr = R0_rect @ Tr_velo_to_cam

    # 扩展为4x4
    R0_tr_4x4 = np.eye(4)
    R0_tr_4x4[:3, :] = R0_tr

    # 计算 lidar2img = cam2img @ R0_tr_4x4
    lidar2img = cam2img @ R0_tr_4x4

    return lidar2img.tolist()

def generate_kitti_infos(data_root, output_path, max_samples=None):
    """生成KITTI数据集标注信息"""
    data_root = Path(data_root)
    train_img_dir = data_root / 'training' / 'image_2'
    train_pc_dir = data_root / 'training' / 'velodyne'
    train_calib_dir = data_root / 'training' / 'calib'
    train_label_dir = data_root / 'training' / 'label_2'

    # 获取所有样本ID
    if train_pc_dir.exists():
        sample_ids = [p.stem for p in sorted(train_pc_dir.glob('*.bin'))]
    else:
        print(f"错误: 点云目录不存在: {train_pc_dir}")
        return

    if max_samples:
        sample_ids = sample_ids[:max_samples]

    print(f"找到 {len(sample_ids)} 个样本 (最多处理 {max_samples if max_samples else '全部'})")

    data_list = []

    for i, sample_id in enumerate(sample_ids):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(sample_ids)}", flush=True)

        data_info = {
            'image_id': i,
            'sample_idx': i,
            'token': sample_id,
            'data_path': str(train_pc_dir / f'{sample_id}.bin'),
        }

        # 图像信息
        img_path = train_img_dir / f'{sample_id}.png'
        if img_path.exists():
            import cv2
            img = cv2.imread(str(img_path))
            if img is not None:
                data_info['images'] = {
                    'CAM2': {
                        'img_path': str(img_path),
                        'width': img.shape[1],
                        'height': img.shape[0],
                    }
                }

        # 标定信息
        calib_path = train_calib_dir / f'{sample_id}.txt'
        if calib_path.exists():
            calib = parse_calib_file(calib_path)

            # P2: 相机投影矩阵 (cam2img)
            if 'P2' in calib:
                data_info['images']['CAM2']['cam2img'] = calib['P2'].tolist()

            # R0_rect: 相机矫正旋转矩阵
            if 'R0_rect' in calib:
                data_info['R0_rect'] = calib['R0_rect'].tolist()

            # Tr_velo_to_cam: 从激光雷达到相机的变换矩阵
            if 'Tr_velo_to_cam' in calib:
                data_info['lidar2cam'] = calib['Tr_velo_to_cam'].tolist()

            # 计算lidar2img投影矩阵
            if 'P2' in calib and 'R0_rect' in calib and 'Tr_velo_to_cam' in calib:
                lidar2img = compute_lidar2img(calib['P2'], calib['R0_rect'], calib['Tr_velo_to_cam'])
                data_info['images']['CAM2']['lidar2img'] = lidar2img

        # 加载点云获取点数
        pc_path = train_pc_dir / f'{sample_id}.bin'
        if pc_path.exists():
            pc = np.fromfile(str(pc_path), dtype=np.float32)
            if len(pc) % 4 == 0:
                num_pts = len(pc) // 4
            else:
                num_pts = len(pc) // 4
            data_info['num_pts'] = num_pts
            data_info['lidar_points'] = {
                'num_pts': num_pts,
                'path': f'{sample_id}.bin'
            }

        # 标注信息（如果有）
        label_path = train_label_dir / f'{sample_id}.txt'
        if label_path.exists():
            data_info['ann_info'] = {'gt_bboxes_3d': [], 'gt_labels_3d': []}

        data_list.append(data_info)

    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为pkl文件
    with open(output_path, 'wb') as f:
        pickle.dump({'data_list': data_list, 'metadata': {'version': '1.0'}}, f)

    print(f"标注文件已保存到: {output_path}")
    print(f"共 {len(data_list)} 个样本")

    return data_list

if __name__ == '__main__':
    data_root = 'data/kitti'
    output_path = 'data/kitti/kitti_infos_train.pkl'

    print("开始生成KITTI标注信息...")
    print(f"数据目录: {data_root}")
    print(f"输出路径: {output_path}")

    # 只处理前20个样本用于测试
    generate_kitti_infos(data_root, output_path, max_samples=20)

    print("完成!")