#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return runpy.run_path(str(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the MVXNet KITTI multimodal config.")
    parser.add_argument(
        "--config",
        default="configs/multimodal/mvxnet_kitti_3class.py",
        help="Path to the MVXNet KITTI config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mvxnet_kitti_3class_inspection",
        help="Directory for the inspection manifest.",
    )
    return parser.parse_args()


def pick_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "experiment_name",
        "work_dir",
        "dataset_type",
        "data_root",
        "class_names",
        "point_cloud_range",
        "voxel_size",
        "input_modality",
        "train_cfg",
        "val_evaluator",
        "test_evaluator",
        "load_from",
    ]
    return {key: cfg.get(key) for key in keys if key in cfg}


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = pick_summary(cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": str(Path(args.config)),
        "summary": summary,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[done] config inspected")
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] experiment_name: {summary.get('experiment_name')}")
    print(f"[done] work_dir: {summary.get('work_dir')}")
    print(f"[done] data_root: {summary.get('data_root')}")
    print(f"[done] load_from: {summary.get('load_from')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())