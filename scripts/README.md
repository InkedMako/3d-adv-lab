# Scripts

This directory contains runnable helpers for the project.

## Current scripts

- `install_ubuntu_env.sh`: install and validate the Ubuntu environment
- `minimal_pointcloud_smoke_test.py`: generate clean, adversarial, and defended point clouds for a smoke test
- `run_pointcloud_pipeline.py`: unified entry script for clean/adversarial/defended/all modes
- `visualize_pointcloud_samples.py`: generate side-by-side 3D and 2D comparison plots from .npy files

## Unified entry examples

Run all modes with default config:

```bash
python scripts/run_pointcloud_pipeline.py --mode all
```

Run only clean samples:

```bash
python scripts/run_pointcloud_pipeline.py --mode clean
```

Run multimodal template settings:

```bash
python scripts/run_pointcloud_pipeline.py --config configs/multimodal/fusion_detection_template.py --mode all
```

## Visualization examples

After running the pipeline, visualize results with comparison plots:

```bash
# Default: reads from outputs/pipeline_test, saves to outputs/vis
python scripts/visualize_pointcloud_samples.py

# Custom paths:
python scripts/visualize_pointcloud_samples.py --input-root outputs/my_experiment --output-dir outputs/comparison
```

This generates:
- `comparison_3d.png`: side-by-side 3D scatter plots
- `comparison_2d_xy.png`: 2D projections on X-Y plane
- `summary.txt`: point count for each sample
