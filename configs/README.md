# Configs

This directory stores experiment configuration templates.

## Suggested structure

- `base/`: shared base configs
- `pointcloud/`: point cloud-only experiments
- `multimodal/`: LiDAR-camera multimodal experiments

## Added templates

- `base/runtime_template.py`: shared runtime, output, attack, and defense conventions
- `pointcloud/classification_template.py`: point cloud classification baseline template
- `pointcloud/detection_template.py`: 3D detection baseline template
- `multimodal/fusion_detection_template.py`: LiDAR-camera fusion detection template

## Recommended starting point

1. If you need the fastest path to first result, start with `pointcloud/classification_template.py`.
2. If your report focus is 3D detection, start with `pointcloud/detection_template.py`.
3. If your mentor requires multimodal direction, start with `multimodal/fusion_detection_template.py`.

Keep dataset paths, model settings, and evaluation settings here instead of hardcoding them in scripts.
