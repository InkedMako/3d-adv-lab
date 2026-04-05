"""3D point cloud detection template for MMDetection3D-style experiments."""

_base_ = ["../base/runtime_template.py"]

experiment_name = "pointcloud_detection_template"
work_dir = "work_dirs/pointcloud_detection_template"


# Choose one and adjust paths
dataset_type = "KittiDataset"
data_root = "data/raw/kitti/"
class_names = ["Car", "Pedestrian", "Cyclist"]


# Keep this minimal and replace with your model config details
model = dict(
    type="VoxelNet",
    data_preprocessor=dict(type="Det3DDataPreprocessor", voxel=True),
    voxel_encoder=dict(type="HardSimpleVFE"),
    middle_encoder=dict(type="SparseEncoder", in_channels=4, sparse_shape=[41, 1600, 1408]),
    backbone=dict(type="SECOND", in_channels=256, layer_nums=[5, 5], layer_strides=[1, 2], out_channels=[128, 256]),
    neck=dict(type="SECONDFPN", in_channels=[128, 256], upsample_strides=[1, 2], out_channels=[256, 256]),
    bbox_head=dict(type="Anchor3DHead", num_classes=len(class_names), in_channels=512),
)


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="kitti_infos_train.pkl",
        data_prefix=dict(pts="training/velodyne", img="training/image_2"),
        metainfo=dict(classes=class_names),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="kitti_infos_val.pkl",
        data_prefix=dict(pts="training/velodyne", img="training/image_2"),
        metainfo=dict(classes=class_names),
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.01))
param_scheduler = [dict(type="CosineAnnealingLR", by_epoch=True, T_max=24, eta_min=1e-5)]

val_evaluator = dict(type="KittiMetric", ann_file=data_root + "kitti_infos_val.pkl", metric="bbox")
test_evaluator = val_evaluator


# Attack/defense test conventions for detection task
artifact_cfg = dict(
    sample_ids=[0, 10, 20, 30],
    save_clean=True,
    save_adversarial=True,
    save_defended=True,
    save_vis=True,
)
