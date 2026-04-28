#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np


REQUIRED_MODULES = ["art", "mmcv", "mmengine", "mmdet", "mmdet3d", "open3d", "torch"]


def check_environment() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"FAIL ({type(exc).__name__}: {exc})"
    return versions


def try_import_open3d():
    try:
        import open3d as o3d

        return o3d
    except Exception as exc:
        print(f"[warn] open3d unavailable: {exc}")
        return None


def make_base_point_cloud(num_points: int = 2048) -> np.ndarray:
    rng = np.random.default_rng(42)
    points = rng.normal(loc=0.0, scale=1.0, size=(num_points, 3)).astype(np.float32)
    return points


def attack_point_cloud(points: np.ndarray, noise_std: float = 0.02, drop_ratio: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(123)
    noisy_points = points + rng.normal(0.0, noise_std, size=points.shape).astype(np.float32)
    keep_mask = rng.random(noisy_points.shape[0]) >= drop_ratio
    attacked = noisy_points[keep_mask]
    if attacked.shape[0] == 0:
        return noisy_points[:1]
    return attacked


def defend_point_cloud(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    o3d = try_import_open3d()
    if o3d is None:
        step = max(1, int(round(1.0 / max(voxel_size, 1e-3))))
        reduced = points[::step]
        return reduced if reduced.shape[0] > 0 else points[:1]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    downsampled = cloud.voxel_down_sample(voxel_size=voxel_size)
    if len(downsampled.points) == 0:
        return points[:1]
    return np.asarray(downsampled.points, dtype=np.float32)


def save_point_cloud(path: Path, points: np.ndarray) -> None:
    o3d = try_import_open3d()
    if o3d is None:
        np.save(path.with_suffix(".npy"), points)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=True)


def render_preview(path: Path, points: np.ndarray) -> None:
    o3d = try_import_open3d()
    if o3d is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=960, height=720)
    vis.add_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(path), do_render=True)
    vis.destroy_window()


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal point cloud smoke test for ART + MMDetection3D project.")
    parser.add_argument("--output-dir", default="outputs/smoke_test", help="Directory for generated files.")
    parser.add_argument("--num-points", type=int, default=2048, help="Number of points in the synthetic cloud.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Gaussian noise used in the attack step.")
    parser.add_argument("--drop-ratio", type=float, default=0.15, help="Point drop ratio used in the attack step.")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size used in the defense step.")
    parser.add_argument("--render", action="store_true", help="Enable 3D visualization preview rendering (requires display, not recommended for headless containers).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    versions = check_environment()
    print("[env] dependency versions")
    for name, version in versions.items():
        print(f"  - {name}: {version}")

    clean_points = make_base_point_cloud(args.num_points)
    attacked_points = attack_point_cloud(clean_points, noise_std=args.noise_std, drop_ratio=args.drop_ratio)
    defended_points = defend_point_cloud(attacked_points, voxel_size=args.voxel_size)

    save_point_cloud(output_dir / "clean.ply", clean_points)
    save_point_cloud(output_dir / "adversarial.ply", attacked_points)
    save_point_cloud(output_dir / "defended.ply", defended_points)

    np.save(output_dir / "clean.npy", clean_points)
    np.save(output_dir / "adversarial.npy", attacked_points)
    np.save(output_dir / "defended.npy", defended_points)

    if args.render:
        try:
            render_preview(output_dir / "clean.png", clean_points)
            render_preview(output_dir / "adversarial.png", attacked_points)
            render_preview(output_dir / "defended.png", defended_points)
            print("[info] preview rendering completed")
        except Exception as exc:
            print(f"[warn] preview rendering failed: {exc}")
    else:
        print("[info] preview rendering skipped (use --render to enable)")

    report = {
        "clean": summarize(clean_points),
        "adversarial": summarize(attacked_points),
        "defended": summarize(defended_points),
        "attack": {
            "noise_std": args.noise_std,
            "drop_ratio": args.drop_ratio,
        },
        "defense": {
            "voxel_size": args.voxel_size,
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[result] saved files")
    print(f"  - {output_dir / 'clean.ply'}")
    print(f"  - {output_dir / 'adversarial.ply'}")
    print(f"  - {output_dir / 'defended.ply'}")
    print(f"  - {output_dir / 'clean.npy'}")
    print(f"  - {output_dir / 'adversarial.npy'}")
    print(f"  - {output_dir / 'defended.npy'}")
    print(f"  - {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())