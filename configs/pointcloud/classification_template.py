"""Point cloud classification template.

This template is framework-agnostic and can drive custom scripts
for ART-based attack and defense evaluation.
"""

_base_ = ["../base/runtime_template.py"]

experiment_name = "pointcloud_classification_template"
work_dir = "work_dirs/pointcloud_classification_template"


# Dataset paths
dataset_type = "ModelNet40Dataset"
data_root = "data/raw/modelnet40/"

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="train",
        with_label=True,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        with_label=True,
    ),
)

test_dataloader = val_dataloader


# Placeholder model settings for a point-cloud classifier
model = dict(
    type="PointCloudClassifier",
    backbone=dict(type="PointNet", feat_dims=1024),
    neck=None,
    head=dict(type="LinearClsHead", num_classes=40, in_channels=1024),
)


# Training schedule
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(optimizer=dict(type="Adam", lr=0.001, weight_decay=1e-4))
param_scheduler = [dict(type="MultiStepLR", by_epoch=True, milestones=[30, 40], gamma=0.1)]


# Evaluation and artifact saving
val_evaluator = dict(type="Accuracy", topk=(1,))
test_evaluator = val_evaluator

artifact_cfg = dict(
    sample_ids=[0, 1, 2, 3, 4],
    save_clean=True,
    save_adversarial=True,
    save_defended=True,
    save_vis=True,
)
