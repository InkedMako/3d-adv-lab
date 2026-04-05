# 3D Adversarial Lab

This workspace is organized for the ART + MMDetection3D project on 3D point cloud and multimodal robustness.

## Layout

- `docs/` project notes, installation, workflow, and direction documents
- `scripts/` runnable helpers and smoke tests
- `configs/` experiment configuration templates
- `data/` dataset staging area
- `work_dirs/` training and inference outputs
- `outputs/` attack, defense, and visualization results

## Quick Start

1. Read [docs/INSTALLATION.md](docs/INSTALLATION.md)
2. Install the environment with `bash scripts/install_ubuntu_env.sh`
3. Run the smoke test with `python scripts/minimal_pointcloud_smoke_test.py --output-dir outputs/smoke_test`
4. Put datasets under `data/`
5. Put experiment configs under `configs/`
