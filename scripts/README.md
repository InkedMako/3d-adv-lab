# Scripts

This directory contains runnable helpers for the project.

## Current scripts

- `install_ubuntu_env.sh`: install and validate the Ubuntu environment
- `minimal_pointcloud_smoke_test.py`: generate clean, adversarial, and defended point clouds for a smoke test
- `run_pointcloud_pipeline.py`: unified entry script for clean/adversarial/defended/all modes
- `visualize_pointcloud_samples.py`: generate side-by-side 3D and 2D comparison plots from .npy files
- `prepare_kitti.py`: validate KITTI directory and generate `kitti_infos_*.pkl`

## KITTI one-command preparation

After copying KITTI files into `data/kitti/`, run:

```bash
python scripts/prepare_kitti.py --data-root data/kitti
```

This script will:
- verify required KITTI folders and ImageSets files
- generate `kitti_infos_train.pkl` and `kitti_infos_val.pkl` when missing
- print clear errors if data is incomplete

Quick check only (without metadata generation):

```bash
python scripts/prepare_kitti.py --data-root data/kitti --skip-generate
```

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

Inspect the formal KITTI MVXNet multimodal config:

```bash
python scripts/inspect_mvxnet_kitti_config.py --config configs/multimodal/mvxnet_kitti_3class.py
```

This generates:
- `outputs/mvxnet_kitti_3class_inspection/manifest.json`
- a compact summary of the experiment name, work dir, dataset, and checkpoint

Launch the MVXNet KITTI training entry:

```bash
python scripts/run_mvxnet_kitti_training.py --config configs/multimodal/mvxnet_kitti_3class.py
```

Dry run only, useful for checking the resolved command:

```bash
python scripts/run_mvxnet_kitti_training.py --config configs/multimodal/mvxnet_kitti_3class.py --dry-run
```

This writes:
- `work_dirs/mvxnet_kitti_3class_train/launch_manifest.json`
- the exact `mim train mmdet3d ...` command used for the run

Run model-level multimodal robustness evaluation (clean/adversarial/defended):

```bash
python scripts/run_multimodal_robustness_eval.py --config configs/multimodal/mvxnet_kitti_3class.py --checkpoint <your_checkpoint_path>
```

This writes:
- `outputs/multimodal_robustness_eval/robustness_report.json`
- generated adversarial/defended temporary lidar and image inputs under `outputs/multimodal_robustness_eval/generated/`

By default, the report includes:
- proxy robustness statistics (prediction count / confidence / recall proxy)
- official KITTI metrics (bbox/bev/3d) for clean, adversarial, and defended branches

If you only want proxy statistics:

```bash
python scripts/run_multimodal_robustness_eval.py --disable-official-kitti-metric
```

Notes:
- The training launcher now enforces project `.venv` Python by default.
- Use `--allow-system-python` only when you intentionally want a non-project interpreter.

## Smoke test examples

Run minimal test without rendering (for headless containers or servers):

```bash
# Default: no visualization, outputs saved to outputs/smoke_test
python scripts/minimal_pointcloud_smoke_test.py

# With rendering (requires display environment):
python scripts/minimal_pointcloud_smoke_test.py --render

# Docker execution (recommended for reproducibility):
docker run --rm -v "${PWD}:/workspace" -w /workspace 3d-adv-lab:cu124 bash -lc \
  "source /workspace/.venv/bin/activate && python scripts/minimal_pointcloud_smoke_test.py"
```

This generates:
- `clean.npy`, `adversarial.npy`, `defended.npy`: raw point cloud arrays
- `clean.ply`, `adversarial.ply`, `defended.ply`: point cloud files in PLY format
- `report.json`: statistics (point count, mean/std for each sample)
- `clean.png`, `adversarial.png`, `defended.png`: optional visualization (with `--render`)

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
