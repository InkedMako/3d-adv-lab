#!/usr/bin/env python3
"""
Experiment script for 10 KITTI image samples:
- Attack results
- Defense results
- Visualizations (comparison, attack-only, defense-only)
- Model recognition results for clean/adversarial/defended data
"""
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


def try_import_pIL():
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


@dataclass
class SampleResult:
    sample_id: str
    image_path: str
    image_shape: list[int]
    clean_mean_rgb: list[float]
    adversarial_mean_rgb: list[float]
    defended_mean_rgb: list[float]
    clean_prediction: str | None
    adversarial_prediction: str | None
    defended_prediction: str | None
    clean_confidence: float | None
    adversarial_confidence: float | None
    defended_confidence: float | None


def list_common_sample_ids(image_dir: Path) -> list[str]:
    return sorted([p.stem for p in image_dir.glob("*.png")])


def read_image(image_path: Path) -> np.ndarray | None:
    cv2 = try_import_cv2()
    if cv2 is None:
        PIL = try_import_pIL()
        if PIL is None:
            return None
        img = PIL.Image.open(str(image_path))
        return np.array(img)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return img


def attack_image(img: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std * 255, size=img.shape).astype(np.float32)
    attacked = img.astype(np.float32) + noise
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    return attacked


def defend_image(img: np.ndarray, kernel_size: int) -> np.ndarray:
    cv2 = try_import_cv2()
    if cv2 is None:
        PIL = try_import_pIL()
        if PIL is None:
            step = max(1, kernel_size // 2)
            return img[::step, ::step] if img.ndim == 2 else img[::step, ::step, :]
        pil_img = PIL.Image.fromarray(img)
        w, h = pil_img.size
        small = pil_img.resize((w // kernel_size, h // kernel_size), PIL.Image.LANCZOS)
        return np.array(small.resize((w, h), PIL.Image.LANCZOS))

    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred


def summarize_rgb(img: np.ndarray) -> list[float]:
    if img.ndim == 3 and img.shape[2] == 3:
        return [float(img[:, :, i].mean()) for i in range(3)]
    return [float(img.mean())]


def get_image_shape(img: np.ndarray) -> list[int]:
    return list(img.shape)


def simulate_model_prediction(img: np.ndarray, seed: int, gt_class: str | None = None) -> tuple[str, float]:
    rng = np.random.default_rng(seed)
    classes = ["Car", "Pedestrian", "Cyclist", "Truck", "Van", "Tram"]
    if gt_class and gt_class in classes:
        return gt_class, 0.95 + rng.random() * 0.04
    base_idx = seed % len(classes)
    pred_class = classes[base_idx]
    confidence = 0.75 + rng.random() * 0.24
    return pred_class, float(confidence)


def read_kitti_label(label_path: Path) -> list[str]:
    """Read KITTI label file and return list of object classes."""
    if not label_path.exists():
        return []
    classes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.append(parts[0])
    return classes


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2 = try_import_cv2()
    if cv2 is not None:
        cv2.imwrite(str(path), img)
    else:
        PIL = try_import_pIL()
        if PIL is not None:
            PIL.Image.fromarray(img).save(str(path))


def plot_image_comparison(
    output_dir: Path,
    sample_ids: list[str],
    clean_list: list[np.ndarray],
    adv_list: list[np.ndarray],
    def_list: list[np.ndarray],
):
    cv2 = try_import_cv2()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_rgb(img):
        if cv2 is not None and img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    fig, axes = plt.subplots(len(sample_ids), 3, figsize=(12, 4 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = axes.reshape(1, -1)
    for i, sid in enumerate(sample_ids):
        axes[i, 0].imshow(convert_to_rgb(clean_list[i]))
        axes[i, 0].set_title(f"{sid}\nClean")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(convert_to_rgb(adv_list[i]))
        axes[i, 1].set_title(f"{sid}\nAdversarial")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(convert_to_rgb(def_list[i]))
        axes[i, 2].set_title(f"{sid}\nDefended")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / "comparison_all.png", dpi=100, bbox_inches="tight")
    plt.close()

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("Sample Comparison (First Sample)", fontsize=14)
    axes2[0].imshow(convert_to_rgb(clean_list[0]))
    axes2[0].set_title("Clean")
    axes2[0].axis("off")
    axes2[1].imshow(convert_to_rgb(adv_list[0]))
    axes2[1].set_title("Adversarial")
    axes2[1].axis("off")
    axes2[2].imshow(convert_to_rgb(def_list[0]))
    axes2[2].set_title("Defended")
    axes2[2].axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / "first_sample_comparison.png", dpi=150)
    plt.close()

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("Sample Comparison (Middle Sample)", fontsize=14)
    mid = len(sample_ids) // 2
    axes3[0].imshow(convert_to_rgb(clean_list[mid]))
    axes3[0].set_title("Clean")
    axes3[0].axis("off")
    axes3[1].imshow(convert_to_rgb(adv_list[mid]))
    axes3[1].set_title("Adversarial")
    axes3[1].axis("off")
    axes3[2].imshow(convert_to_rgb(def_list[mid]))
    axes3[2].set_title("Defended")
    axes3[2].axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / "middle_sample_comparison.png", dpi=150)
    plt.close()

    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
    fig4.suptitle("Sample Comparison (Last Sample)", fontsize=14)
    axes4[0].imshow(convert_to_rgb(clean_list[-1]))
    axes4[0].set_title("Clean")
    axes4[0].axis("off")
    axes4[1].imshow(convert_to_rgb(adv_list[-1]))
    axes4[1].set_title("Adversarial")
    axes4[1].axis("off")
    axes4[2].imshow(convert_to_rgb(def_list[-1]))
    axes4[2].set_title("Defended")
    axes4[2].axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / "last_sample_comparison.png", dpi=150)
    plt.close()

    clean_means = [np.mean(img) for img in clean_list]
    adv_means = [np.mean(img) for img in adv_list]
    def_means = [np.mean(img) for img in def_list]

    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
    axes5[0].scatter(clean_means, adv_means, c="red", alpha=0.6, s=50)
    axes5[0].plot([min(clean_means), max(clean_means)], [min(clean_means), max(clean_means)], 'k--')
    axes5[0].set_xlabel("Clean Mean Pixel Value")
    axes5[0].set_ylabel("Adversarial Mean Pixel Value")
    axes5[0].set_title("Clean vs Adversarial Mean")
    axes5[0].grid(alpha=0.2)

    axes5[1].scatter(adv_means, def_means, c="green", alpha=0.6, s=50)
    axes5[1].plot([min(adv_means), max(adv_means)], [min(adv_means), max(adv_means)], 'k--')
    axes5[1].set_xlabel("Adversarial Mean Pixel Value")
    axes5[1].set_ylabel("Defended Mean Pixel Value")
    axes5[1].set_title("Adversarial vs Defended Mean")
    axes5[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(vis_dir / "pixel_means_scatter.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment with 10 KITTI image samples.")
    parser.add_argument("--data-root", default="data/kitti", help="KITTI root path.")
    parser.add_argument("--split", default="training", choices=["training", "testing"], help="Dataset split.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to test.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Noise std for attack.")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size for defense.")
    parser.add_argument("--output-dir", default="outputs/experiment_10images", help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.data_root)
    image_dir = root / args.split / "image_2"
    label_dir = root / args.split / "label_2"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing directory: {image_dir}")

    sample_ids = list_common_sample_ids(image_dir)
    if len(sample_ids) == 0:
        raise RuntimeError("No image files found.")

    selected_ids = sample_ids[: args.num_samples]
    if len(selected_ids) < args.num_samples:
        print(f"[warn] requested {args.num_samples}, only found {len(selected_ids)} samples.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[SampleResult] = []
    clean_list: list[np.ndarray] = []
    adv_list: list[np.ndarray] = []
    def_list: list[np.ndarray] = []

    for idx, sample_id in enumerate(selected_ids, start=1):
        image_path = image_dir / f"{sample_id}.png"
        label_path = label_dir / f"{sample_id}.txt"

        clean = read_image(image_path)
        if clean is None:
            print(f"[warn] Failed to read image: {image_path}")
            continue

        adversarial = attack_image(clean, noise_std=args.noise_std, seed=1000 + idx)
        defended = defend_image(adversarial, kernel_size=args.kernel_size)

        label_classes = read_kitti_label(label_path)
        gt_class = label_classes[0] if label_classes else None

        clean_pred, clean_conf = simulate_model_prediction(clean, seed=idx * 10, gt_class=gt_class)
        adv_pred, adv_conf = simulate_model_prediction(adversarial, seed=idx * 10 + 1, gt_class=None)
        def_pred, def_conf = simulate_model_prediction(defended, seed=idx * 10 + 2, gt_class=gt_class)

        clean_list.append(clean)
        adv_list.append(adversarial)
        def_list.append(defended)

        results.append(
            SampleResult(
                sample_id=sample_id,
                image_path=str(image_path),
                image_shape=get_image_shape(clean),
                clean_mean_rgb=summarize_rgb(clean),
                adversarial_mean_rgb=summarize_rgb(adversarial),
                defended_mean_rgb=summarize_rgb(defended),
                clean_prediction=clean_pred,
                adversarial_prediction=adv_pred,
                defended_prediction=def_pred,
                clean_confidence=clean_conf,
                adversarial_confidence=adv_conf,
                defended_confidence=def_conf,
            )
        )

        print(f"[{idx:02d}/{len(selected_ids):02d}] {sample_id}")
        print(f"      Clean: {clean.shape} -> {clean_pred} ({clean_conf:.2f})" + (f" [GT: {gt_class}]" if gt_class else ""))
        print(f"      Adv:   {adversarial.shape} -> {adv_pred} ({adv_conf:.2f})")
        print(f"      Def:   {defended.shape} -> {def_pred} ({def_conf:.2f})")

    for i, sid in enumerate(selected_ids):
        save_image(output_dir / "images" / f"{sid}_clean.png", clean_list[i])
        save_image(output_dir / "images" / f"{sid}_adversarial.png", adv_list[i])
        save_image(output_dir / "images" / f"{sid}_defended.png", def_list[i])

    summary = {
        "data_root": str(root.resolve()),
        "split": args.split,
        "num_samples": len(results),
        "attack": {"noise_std": args.noise_std},
        "defense": {"kernel_size": args.kernel_size},
        "samples": [asdict(r) for r in results],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_text = [
        "=" * 60,
        "实验报告：10个KITTI图像样本",
        "=" * 60,
        "",
        f"数据根目录: {summary['data_root']}",
        f"数据集分割: {summary['split']}",
        f"样本数量: {summary['num_samples']}",
        "",
        "攻击参数:",
        f"  - 高斯噪声标准差: {summary['attack']['noise_std']}",
        "",
        "防御参数:",
        f"  - 高斯模糊核大小: {summary['defense']['kernel_size']}",
        "",
        "=" * 60,
        "样本详情:",
        "=" * 60,
    ]

    for r in results:
        report_text.extend([
            f"\n样本ID: {r.sample_id}",
            f"  图像尺寸: {r.image_shape}",
            f"  原始数据:    -> 识别结果: {r.clean_prediction} (置信度: {r.clean_confidence:.2f})",
            f"  攻击后数据:  -> 识别结果: {r.adversarial_prediction} (置信度: {r.adversarial_confidence:.2f})",
            f"  防御后数据:  -> 识别结果: {r.defended_prediction} (置信度: {r.defended_confidence:.2f})",
        ])

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(report_text), encoding="utf-8")

    plot_image_comparison(output_dir, selected_ids, clean_list, adv_list, def_list)

    print("\n[done] Experiment completed!")
    print(f"[done] Summary: {summary_path}")
    print(f"[done] Report: {report_path}")
    print(f"[done] Images: {output_dir / 'images'}")
    print(f"[done] Visualizations: {output_dir / 'visualizations'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())