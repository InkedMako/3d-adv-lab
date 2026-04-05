# 3D Adversarial Lab (English Guide)

This repository is organized for adversarial robustness experiments with ART + MMDetection3D, currently focused on multimodal 3D perception (LiDAR point clouds + camera images).

## 1. Repository Structure

3d-adv-lab/
- configs/
- data/
- docs/
- outputs/
- scripts/
- work_dirs/
- Dockerfile
- environment.yml
- requirements.txt
- README.md
- README_EN.md

## 2. Folder Purposes and Required File Types

### 2.1 configs/

Purpose: experiment configuration files for datasets, models, attacks, defenses, and runtime parameters.

Subfolders:
- configs/base/: shared runtime templates
- configs/pointcloud/: point-cloud task configs
- configs/multimodal/: multimodal task configs

Required file types:
- Required: .py (config scripts)
- Optional: .md (folder notes)

Naming suggestions:
- *_template.py for templates
- *_baseline.py for reproducible baselines
- *_expXX.py for experiment variants

### 2.2 data/

Purpose: dataset storage and preprocessing artifacts.

Subfolders:
- data/raw/: immutable raw downloaded data
- data/processed/: converted/sampled/indexed data
- data/downloads/: downloaded archives cache

Expected file types:
- Raw data: .bin, .pcd, .ply, .jpg, .png, .txt
- Labels/indexes: .txt, .json, .pkl
- Archives: .zip, .tar, .gz

Note:
- Real large datasets are ignored by Git by default. Only folder structure and notes are tracked.

### 2.3 docs/

Purpose: installation guides, workflows, methodology notes, and troubleshooting docs.

Current docs:
- docs/INSTALLATION.md
- docs/POINTCLOUD_TESTING_GUIDE.md
- docs/DATASET_ATTACK_DEFENSE_WORKFLOW.md
- docs/MULTIMODAL_DIRECTION.md

Required file types:
- Required: .md
- Optional: .pdf, .docx (for final reports)

### 2.4 outputs/

Purpose: experiment outputs (samples, metrics, visualizations).

Subfolders:
- outputs/pipeline_test/: outputs from the unified pipeline script
- outputs/smoke_test/: outputs from minimum smoke test
- outputs/vis/: generated visualization files

Expected file types:
- Samples: .npy
- Reports: .json, .txt
- Visualizations: .png, .jpg

Current examples:
- clean.npy, adversarial.npy, defended.npy
- report.json
- comparison_3d.png, comparison_2d_xy.png, summary.txt

Note:
- Real experiment artifacts are ignored by Git by default. Only folder skeleton and notes are tracked.

### 2.5 scripts/

Purpose: runnable scripts (environment setup, tests, pipelines, visualization).

Current scripts:
- scripts/install_ubuntu_env.sh
- scripts/minimal_pointcloud_smoke_test.py
- scripts/run_pointcloud_pipeline.py
- scripts/visualize_pointcloud_samples.py

Required file types:
- Python scripts: .py
- Shell scripts: .sh
- Script notes: .md

Script quality suggestions:
- Support CLI arguments
- Write outputs to outputs/ or work_dirs/
- Provide clear error messages

### 2.6 work_dirs/

Purpose: training/inference working directories.

Typical contents:
- Logs: .log, .json
- Model checkpoints: .pth, .pt, .ckpt
- Temporary files from training/inference

Required file types:
- Framework-generated files, commonly .pth/.log/.json

Note:
- work_dirs artifacts are ignored by Git by default; only folder skeleton is tracked.

## 3. Root-Level Files

- Dockerfile: containerized runtime definition
- environment.yml: Conda environment definition
- requirements.txt: pip dependency list
- .gitignore: ignore policy (keeps data/outputs/work_dirs skeleton only)
- .gitattributes: Git LFS tracking rules

## 4. Placement Rules (Project Conventions)

1. Put new datasets into data/raw/ first, then generate processed files under data/processed/.
2. Store all experiment outputs under outputs/ or work_dirs/.
3. Keep all experiment settings in configs/ and load them via script arguments.
4. Keep project documents under docs/ and ensure paths are consistent with code.

## 5. Quick Start

1. Read docs/INSTALLATION.md and finish environment setup.
2. Run scripts/minimal_pointcloud_smoke_test.py to verify the basic pipeline.
3. Run scripts/run_pointcloud_pipeline.py to generate clean/adv/defended samples.
4. Run scripts/visualize_pointcloud_samples.py to generate comparison figures.

## 6. Environment Rebuild and Commit Policy

Do not commit these folders to GitHub:
- .venv/
- real datasets under data/
- real experiment artifacts under outputs/
- large checkpoints/logs under work_dirs/

Recommended workflow:
1. Commit code, configs, docs, and environment definition files (requirements.txt / environment.yml / Dockerfile).
2. Rebuild environments locally or in containers, instead of uploading .venv.
3. Keep only folder skeleton files (.gitkeep/README) under data/outputs/work_dirs.

Rebuild examples:
- Docker: use scripts/install_ubuntu_env.sh inside container runtime.
- Local Python: recreate the environment from requirements.txt or environment.yml.
