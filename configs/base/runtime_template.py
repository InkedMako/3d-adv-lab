"""Shared runtime template for all experiments.

Copy this file and adjust values per experiment.
"""

default_scope = "mmdet3d"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="Det3DVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_level = "INFO"
load_from = None
resume = False

randomness = dict(seed=42, deterministic=False)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")


# Workspace conventions
work_dir = "work_dirs/template"


# Shared output conventions for clean/adv/defense artifacts
output_cfg = dict(
    clean_dir="outputs/clean",
    adversarial_dir="outputs/adversarial",
    defended_dir="outputs/defended",
    vis_dir="outputs/vis",
)


# Optional settings for attack/defense scripts
attack_cfg = dict(type="jitter_drop", noise_std=0.02, drop_ratio=0.15)
defense_cfg = dict(type="voxel_downsample", voxel_size=0.05)
