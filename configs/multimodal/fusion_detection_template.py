"""LiDAR-camera fusion detection template.

Use this as a starting point for multimodal robustness experiments.
"""

_base_ = ["../base/runtime_template.py"]

experiment_name = "multimodal_fusion_detection_template"
work_dir = "work_dirs/multimodal_fusion_detection_template"


# Suggest nuScenes for multimodal experiments
dataset_type = "NuScenesDataset"
data_root = "data/raw/nuscenes/"
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


# Placeholder fusion model (replace with your selected architecture)
model = dict(
    type="BEVFusion",
    data_preprocessor=dict(type="Det3DDataPreprocessor", voxel=True),
    pts_voxel_encoder=dict(type="HardSimpleVFE"),
    pts_middle_encoder=dict(type="SparseEncoder", in_channels=5, sparse_shape=[1440, 1440, 41]),
    pts_backbone=dict(type="SECOND", in_channels=256, layer_nums=[5, 5], layer_strides=[1, 2], out_channels=[128, 256]),
    img_backbone=dict(type="ResNet", depth=50, num_stages=4, out_indices=(1, 2, 3)),
    img_neck=dict(type="FPN", in_channels=[512, 1024, 2048], out_channels=256, num_outs=4),
    fusion_layer=dict(type="ConvFuser", in_channels=[256, 256], out_channels=256),
    bbox_head=dict(type="CenterHead", in_channels=256, tasks=[dict(num_class=len(class_names), class_names=class_names)]),
)


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="nuscenes_infos_train.pkl",
        data_prefix=dict(pts="samples/LIDAR_TOP", img="samples/CAM_FRONT"),
        modality=dict(use_lidar=True, use_camera=True),
        metainfo=dict(classes=class_names),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="nuscenes_infos_val.pkl",
        data_prefix=dict(pts="samples/LIDAR_TOP", img="samples/CAM_FRONT"),
        modality=dict(use_lidar=True, use_camera=True),
        metainfo=dict(classes=class_names),
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=20, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.01))
param_scheduler = [dict(type="CosineAnnealingLR", by_epoch=True, T_max=20, eta_min=1e-6)]

val_evaluator = dict(type="NuScenesMetric", ann_file=data_root + "nuscenes_infos_val.pkl", metric="bbox")
test_evaluator = val_evaluator


# Multimodal robustness comparison knobs
robustness_cfg = dict(
    evaluate_clean=True,
    evaluate_lidar_attack=True,
    evaluate_camera_attack=True,
    evaluate_joint_attack=False,
    evaluate_defense=True,
)

artifact_cfg = dict(
    sample_ids=[0, 50, 100, 150],
    save_clean=True,
    save_adversarial=True,
    save_defended=True,
    save_vis=True,
)
