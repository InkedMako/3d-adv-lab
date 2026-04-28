#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def try_import_cv2():
    try:
        import cv2

        return cv2
    except Exception:
        return None


def try_import_open3d():
    try:
        import open3d as o3d

        return o3d
    except Exception:
        return None


@dataclass
class SampleResult:
    sample_id: str
    image_path: str
    pointcloud_path: str
    image_height: int
    image_width: int
    clean_points: int
    adversarial_points: int
    defended_points: int
    clean_mean_xyz: list[float]
    adversarial_mean_xyz: list[float]
    defended_mean_xyz: list[float]


def list_common_sample_ids(image_dir: Path, lidar_dir: Path) -> list[str]:
    image_ids = {p.stem for p in image_dir.glob("*.png")}
    lidar_ids = {p.stem for p in lidar_dir.glob("*.bin")}
    return sorted(image_ids.intersection(lidar_ids))


def read_points(bin_path: Path) -> np.ndarray:
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Unexpected KITTI .bin shape: {bin_path}")
    return arr.reshape(-1, 4)[:, :3]


def read_image_shape(image_path: Path) -> tuple[int, int]:
    cv2 = try_import_cv2()
    if cv2 is None:
        return -1, -1
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return -1, -1
    h, w = img.shape[:2]
    return int(h), int(w)


def attack_point_cloud(points: np.ndarray, noise_std: float, drop_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy_points = points + rng.normal(0.0, noise_std, size=points.shape).astype(np.float32)
    keep_mask = rng.random(noisy_points.shape[0]) >= drop_ratio
    attacked = noisy_points[keep_mask]
    if attacked.shape[0] == 0:
        return noisy_points[:1]
    return attacked


def defend_point_cloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    o3d = try_import_open3d()
    if o3d is None:
        step = max(2, int(round(1.0 / max(voxel_size, 1e-6))))
        reduced = points[::step]
        return reduced if reduced.shape[0] > 0 else points[:1]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    downsampled = cloud.voxel_down_sample(voxel_size=voxel_size)
    arr = np.asarray(downsampled.points, dtype=np.float32)
    if arr.shape[0] == 0:
        return points[:1]
    return arr


def summarize_xyz(points: np.ndarray) -> list[float]:
    return [float(points[:, i].mean()) for i in range(3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test for KITTI image + point cloud pairs.")
    parser.add_argument("--data-root", default="data/kitti", help="KITTI root path.")
    parser.add_argument("--split", default="training", choices=["training", "testing"], help="Dataset split.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of matched samples to test.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Gaussian noise std for attack.")
    parser.add_argument("--drop-ratio", type=float, default=0.15, help="Point drop ratio for attack.")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size for defense.")
    parser.add_argument(
        "--output-dir",
        default="outputs/kitti_20images",
        help="Directory to store test outputs.",
    )
    return parser.parse_args()


def save_visualizations(
    vis_dir: Path,
    sample_ids: list[str],
    clean_list: list[np.ndarray],
    adv_list: list[np.ndarray],
    def_list: list[np.ndarray],
) -> None:
    vis_dir.mkdir(parents=True, exist_ok=True)
    legacy_plot = vis_dir / "first_sample_xy.png"
    if legacy_plot.exists():
        legacy_plot.unlink()

    clean_counts = [arr.shape[0] for arr in clean_list]
    adv_counts = [arr.shape[0] for arr in adv_list]
    def_counts = [arr.shape[0] for arr in def_list]

    x = np.arange(len(sample_ids))
    plt.figure(figsize=(12, 5))
    plt.plot(x, clean_counts, marker="o", label="clean")
    plt.plot(x, adv_counts, marker="o", label="adversarial")
    plt.plot(x, def_counts, marker="o", label="defended")
    plt.xticks(x, sample_ids, rotation=45)
    plt.xlabel("sample id")
    plt.ylabel("num points")
    plt.title(f"Point Counts per Sample (n={len(sample_ids)})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "point_counts.png", dpi=150)
    plt.close()

    # Use first/middle/last samples so different experiment sizes are visually distinguishable.
    rep_indices = sorted({0, len(sample_ids) // 2, len(sample_ids) - 1})
    fig, axes = plt.subplots(len(rep_indices), 3, figsize=(15, 4 * len(rep_indices)), sharex=True, sharey=True)
    if len(rep_indices) == 1:
        axes = np.array([axes])

    for row, idx in enumerate(rep_indices):
        sid = sample_ids[idx]
        row_axes = axes[row]
        row_axes[0].scatter(clean_list[idx][:, 0], clean_list[idx][:, 1], s=0.2)
        row_axes[0].set_title(f"clean {sid}")
        row_axes[1].scatter(adv_list[idx][:, 0], adv_list[idx][:, 1], s=0.2)
        row_axes[1].set_title(f"adversarial {sid}")
        row_axes[2].scatter(def_list[idx][:, 0], def_list[idx][:, 1], s=0.2)
        row_axes[2].set_title(f"defended {sid}")
        for ax in row_axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.2)

    fig.suptitle("Representative Sample XY Projections (first/middle/last)")
    fig.tight_layout()
    fig.savefig(vis_dir / "representative_samples_xy.png", dpi=150)
    plt.close(fig)

    summary_txt = (
        f"samples: {len(sample_ids)}\n"
        f"id range: {sample_ids[0]} -> {sample_ids[-1]}\n"
        f"avg clean: {np.mean(clean_counts):.2f}\n"
        f"avg adversarial: {np.mean(adv_counts):.2f}\n"
        f"avg defended: {np.mean(def_counts):.2f}\n"
    )
    (vis_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.data_root)
    image_dir = root / args.split / "image_2"
    lidar_dir = root / args.split / "velodyne"

    if not image_dir.exists() or not lidar_dir.exists():
        raise FileNotFoundError(f"Missing required directories: {image_dir}, {lidar_dir}")

    sample_ids = list_common_sample_ids(image_dir, lidar_dir)
    if len(sample_ids) == 0:
        raise RuntimeError("No matched image/pointcloud pairs found.")

    selected_ids = sample_ids[: args.num_samples]
    if len(selected_ids) < args.num_samples:
        print(f"[warn] requested {args.num_samples}, only found {len(selected_ids)} matched pairs.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    results: list[SampleResult] = []
    clean_list: list[np.ndarray] = []
    adv_list: list[np.ndarray] = []
    def_list: list[np.ndarray] = []

    for idx, sample_id in enumerate(selected_ids, start=1):
        image_path = image_dir / f"{sample_id}.png"
        lidar_path = lidar_dir / f"{sample_id}.bin"

        clean = read_points(lidar_path)
        adversarial = attack_point_cloud(
            clean,
            noise_std=args.noise_std,
            drop_ratio=args.drop_ratio,
            seed=1000 + idx,
        )
        defended = defend_point_cloud(adversarial, voxel_size=args.voxel_size)

        clean_list.append(clean)
        adv_list.append(adversarial)
        def_list.append(defended)

        h, w = read_image_shape(image_path)
        results.append(
            SampleResult(
                sample_id=sample_id,
                image_path=str(image_path),
                pointcloud_path=str(lidar_path),
                image_height=h,
                image_width=w,
                clean_points=int(clean.shape[0]),
                adversarial_points=int(adversarial.shape[0]),
                defended_points=int(defended.shape[0]),
                clean_mean_xyz=summarize_xyz(clean),
                adversarial_mean_xyz=summarize_xyz(adversarial),
                defended_mean_xyz=summarize_xyz(defended),
            )
        )
        print(
            f"[{idx:02d}/{len(selected_ids):02d}] {sample_id} | "
            f"clean={clean.shape[0]} adv={adversarial.shape[0]} def={defended.shape[0]}"
        )

    summary = {
        "data_root": str(root.resolve()),
        "split": args.split,
        "num_requested": args.num_samples,
        "num_tested": len(results),
        "attack": {"noise_std": args.noise_std, "drop_ratio": args.drop_ratio},
        "defense": {"voxel_size": args.voxel_size},
        "avg_clean_points": float(np.mean([r.clean_points for r in results])),
        "avg_adversarial_points": float(np.mean([r.adversarial_points for r in results])),
        "avg_defended_points": float(np.mean([r.defended_points for r in results])),
        "samples": [asdict(r) for r in results],
    }

    summary_path = vis_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ids_path = vis_dir / "tested_ids.txt"
    ids_path.write_text("\n".join([r.sample_id for r in results]) + "\n", encoding="utf-8")

    np.save(out_dir / "clean.npy", np.array(clean_list, dtype=object), allow_pickle=True)
    np.save(out_dir / "adversarial.npy", np.array(adv_list, dtype=object), allow_pickle=True)
    np.save(out_dir / "defended.npy", np.array(def_list, dtype=object), allow_pickle=True)

    save_visualizations(
        vis_dir=vis_dir,
        sample_ids=[r.sample_id for r in results],
        clean_list=clean_list,
        adv_list=adv_list,
        def_list=def_list,
    )

    print("[done] batch test finished")
    print(f"[done] summary: {summary_path}")
    print(f"[done] ids: {ids_path}")
    print(f"[done] npy: {out_dir / 'clean.npy'}")
    print(f"[done] npy: {out_dir / 'adversarial.npy'}")
    print(f"[done] npy: {out_dir / 'defended.npy'}")
    print(f"[done] vis: {vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
