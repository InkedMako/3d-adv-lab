#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def plot_3d_scatter(
    ax,
    points: np.ndarray,
    title: str,
    color: str = "blue",
) -> None:
    if points is None or points.shape[0] == 0:
        ax.set_title(f"{title} (empty)")
        return

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=color, alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title} (n={points.shape[0]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize point cloud samples side by side.")
    parser.add_argument("--input-root", default="outputs/pipeline_test", help="Root directory with clean/adversarial/defended .npy files.")
    parser.add_argument("--output-dir", default="outputs/vis", help="Output directory for visualizations.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = load_npy(input_root / "clean.npy")
    adversarial = load_npy(input_root / "adversarial.npy")
    defended = load_npy(input_root / "defended.npy")

    if clean is None and adversarial is None and defended is None:
        print(f"[error] no point cloud files found in {input_root}")
        return 1

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Point Cloud Samples: Clean / Adversarial / Defended", fontsize=16, y=0.98)

    if clean is not None:
        ax1 = fig.add_subplot(131, projection="3d")
        plot_3d_scatter(ax1, clean, "Clean", color="blue")

    if adversarial is not None:
        ax2 = fig.add_subplot(132, projection="3d")
        plot_3d_scatter(ax2, adversarial, "Adversarial", color="red")

    if defended is not None:
        ax3 = fig.add_subplot(133, projection="3d")
        plot_3d_scatter(ax3, defended, "Defended", color="green")

    plt.tight_layout()
    output_fig = output_dir / "comparison_3d.png"
    plt.savefig(str(output_fig), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_fig}")

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("2D Projections (X-Y plane)", fontsize=16)

    if clean is not None:
        axes[0].scatter(clean[:, 0], clean[:, 1], s=1, c="blue", alpha=0.6)
        axes[0].set_title(f"Clean (n={clean.shape[0]})")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

    if adversarial is not None:
        axes[1].scatter(adversarial[:, 0], adversarial[:, 1], s=1, c="red", alpha=0.6)
        axes[1].set_title(f"Adversarial (n={adversarial.shape[0]})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")

    if defended is not None:
        axes[2].scatter(defended[:, 0], defended[:, 1], s=1, c="green", alpha=0.6)
        axes[2].set_title(f"Defended (n={defended.shape[0]})")
        axes[2].set_xlabel("X")
        axes[2].set_ylabel("Y")

    plt.tight_layout()
    output_fig2 = output_dir / "comparison_2d_xy.png"
    plt.savefig(str(output_fig2), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_fig2}")

    summary = {
        "clean": clean.shape if clean is not None else None,
        "adversarial": adversarial.shape if adversarial is not None else None,
        "defended": defended.shape if defended is not None else None,
    }
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join([f"{k}: {v}" for k, v in summary.items()]),
        encoding="utf-8",
    )
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
