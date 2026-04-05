#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return runpy.run_path(str(config_path))


def try_import_open3d():
    try:
        import open3d as o3d

        return o3d
    except Exception:
        return None


def generate_clean(num_points: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(num_points, 3)).astype(np.float32)


def generate_adversarial(points: np.ndarray, noise_std: float, drop_ratio: float, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = points + rng.normal(0.0, noise_std, size=points.shape).astype(np.float32)
    mask = rng.random(noisy.shape[0]) >= drop_ratio
    adv = noisy[mask]
    if adv.shape[0] == 0:
        return noisy[:1]
    return adv


def generate_defended(points: np.ndarray, voxel_size: float) -> np.ndarray:
    o3d = try_import_open3d()
    if o3d is None:
        step = max(2, int(round(1.0 / max(voxel_size, 1e-6))))
        return points[::step] if points.shape[0] > 1 else points

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    downsampled = cloud.voxel_down_sample(voxel_size=voxel_size)
    arr = np.asarray(downsampled.points, dtype=np.float32)
    if arr.shape[0] == 0:
        return points[:1]
    return arr


def summarize(points: np.ndarray) -> dict[str, float | int]:
    return {
        "num_points": int(points.shape[0]),
        "mean_x": float(points[:, 0].mean()),
        "mean_y": float(points[:, 1].mean()),
        "mean_z": float(points[:, 2].mean()),
        "std_x": float(points[:, 0].std()),
        "std_y": float(points[:, 1].std()),
        "std_z": float(points[:, 2].std()),
    }


def save_cloud(prefix: Path, points: np.ndarray) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_suffix(".npy"), points)

    o3d = try_import_open3d()
    if o3d is None:
        return

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(prefix.with_suffix(".ply")), cloud, write_ascii=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified entry for clean/adversarial/defended point cloud samples.")
    parser.add_argument(
        "--config",
        default="configs/pointcloud/classification_template.py",
        help="Experiment config file path.",
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "adversarial", "defended", "all"],
        default="all",
        help="Pipeline mode.",
    )
    parser.add_argument("--num-points", type=int, default=2048, help="Number of points for synthetic clean sample.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic clean sample generation.")
    parser.add_argument("--output-root", default="outputs/pipeline", help="Root output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))

    attack_cfg = cfg.get("attack_cfg", {})
    defense_cfg = cfg.get("defense_cfg", {})

    noise_std = float(attack_cfg.get("noise_std", 0.02))
    drop_ratio = float(attack_cfg.get("drop_ratio", 0.15))
    voxel_size = float(defense_cfg.get("voxel_size", 0.05))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    clean_points = generate_clean(args.num_points, seed=args.seed)
    adv_points = generate_adversarial(clean_points, noise_std=noise_std, drop_ratio=drop_ratio)
    defended_points = generate_defended(adv_points, voxel_size=voxel_size)

    report: dict[str, Any] = {
        "config": str(args.config),
        "mode": args.mode,
        "attack_cfg": {"noise_std": noise_std, "drop_ratio": drop_ratio},
        "defense_cfg": {"voxel_size": voxel_size},
    }

    if args.mode in ("clean", "all"):
        save_cloud(output_root / "clean", clean_points)
        report["clean"] = summarize(clean_points)

    if args.mode in ("adversarial", "all"):
        save_cloud(output_root / "adversarial", adv_points)
        report["adversarial"] = summarize(adv_points)

    if args.mode in ("defended", "all"):
        save_cloud(output_root / "defended", defended_points)
        report["defended"] = summarize(defended_points)

    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved outputs in:")
    print(output_root)
    print("Report:")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())