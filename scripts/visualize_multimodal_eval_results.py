#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_points(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(str(bin_path), dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return pts.reshape(-1, 4)


def plot_bev(ax, pts: np.ndarray, title: str, color: str) -> None:
    if pts.shape[0] == 0:
        ax.set_title(f"{title} (empty)")
        ax.set_xlim(-40, 40)
        ax.set_ylim(0, 80)
        ax.grid(True, alpha=0.2)
        return

    mask = (
        (pts[:, 0] >= -40)
        & (pts[:, 0] <= 40)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] <= 80)
    )
    pts = pts[mask]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c=color, alpha=0.6)
    ax.set_title(f"{title} (n={pts.shape[0]})")
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 80)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate visualization panels for multimodal robustness evaluation outputs.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory containing robustness_report.json and generated/.")
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum number of samples to visualize.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_path = output_dir / "robustness_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    data_root = Path(report["data_root"])

    sample_ids = [x["sample_id"] for x in report["per_sample"]["clean"]][: args.max_samples]

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in sample_ids:
        clean_img = data_root / "training" / "image_2" / f"{sample_id}.png"
        adv_img = output_dir / "generated" / "adversarial" / "image_2" / f"{sample_id}.png"
        def_img = output_dir / "generated" / "defended" / "image_2" / f"{sample_id}.png"

        clean_lidar = data_root / "training" / "velodyne" / f"{sample_id}.bin"
        adv_lidar = output_dir / "generated" / "adversarial" / "velodyne" / f"{sample_id}.bin"
        def_lidar = output_dir / "generated" / "defended" / "velodyne" / f"{sample_id}.bin"

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f"Sample {sample_id}: Clean / Adversarial / Defended", fontsize=14)

        axes[0, 0].imshow(read_image(clean_img))
        axes[0, 0].set_title("Image Clean")
        axes[0, 1].imshow(read_image(adv_img))
        axes[0, 1].set_title("Image Adversarial")
        axes[0, 2].imshow(read_image(def_img))
        axes[0, 2].set_title("Image Defended")
        for j in range(3):
            axes[0, j].axis("off")

        plot_bev(axes[1, 0], load_points(clean_lidar), "LiDAR Clean (BEV)", "royalblue")
        plot_bev(axes[1, 1], load_points(adv_lidar), "LiDAR Adversarial (BEV)", "crimson")
        plot_bev(axes[1, 2], load_points(def_lidar), "LiDAR Defended (BEV)", "forestgreen")

        fig.tight_layout()
        fig.savefig(vis_dir / f"{sample_id}_panel.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    summary_path = vis_dir / "README.txt"
    summary_path.write_text(
        "Generated per-sample visualization panels.\n"
        "Each panel: top row image clean/adv/def, bottom row LiDAR BEV clean/adv/def.\n",
        encoding="utf-8",
    )

    print(f"[done] visualizations saved to {vis_dir}")
    print(f"[done] sample panels: {len(sample_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
