#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import traceback
from pathlib import Path
from typing import Any

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


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return runpy.run_path(str(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal clean/adversarial/defended robustness evaluation.")
    parser.add_argument("--config", default="configs/multimodal/mvxnet_kitti_3class.py", help="Model config path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. If omitted, use load_from from config.")
    parser.add_argument("--ann-file", default=None, help="KITTI info file path. If omitted, infer from val dataloader.")
    parser.add_argument("--data-root", default=None, help="Dataset root path. If omitted, infer from config.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples for evaluation.")
    parser.add_argument("--cam-type", default="CAM2", help="Camera key for multimodal inference.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold for counting predictions.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Lidar Gaussian noise std for attack.")
    parser.add_argument("--drop-ratio", type=float, default=0.15, help="Lidar point drop ratio for attack.")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Lidar defense voxel size.")
    parser.add_argument("--img-noise-std", type=float, default=8.0, help="Image Gaussian noise std in pixel space.")
    parser.add_argument("--img-blur-ksize", type=int, default=5, help="Image blur kernel size for defense.")
    parser.add_argument("--output-dir", default="outputs/multimodal_robustness_eval", help="Output directory.")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Only check required runtime dependencies and exit.",
    )
    parser.add_argument(
        "--disable-official-kitti-metric",
        action="store_true",
        help="Disable official KITTI metric computation and only keep proxy statistics.",
    )
    return parser.parse_args()


def import_mmengine():
    try:
        import mmengine  # type: ignore

        return mmengine
    except Exception as exc:
        raise RuntimeError(
            "mmengine is required for this script. "
            "Please run inside the project environment, for example:\n"
            "  .\\.venv\\Scripts\\python.exe scripts/run_multimodal_robustness_eval.py --help"
        ) from exc


def check_runtime_dependencies() -> tuple[bool, dict[str, str]]:
    modules = ["mmengine", "mmdet3d", "torch", "numpy"]
    report: dict[str, str] = {}
    ok = True
    for module_name in modules:
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            ok = False
            report[module_name] = f"FAIL ({type(exc).__name__}: {exc})"
    return ok, report


def resolve_data_root(cfg: dict[str, Any], data_root_arg: str | None) -> Path:
    if data_root_arg:
        return Path(data_root_arg)
    data_root = cfg.get("data_root")
    if not data_root:
        raise ValueError("Cannot resolve data_root from config.")
    return Path(str(data_root))


def resolve_ann_file(cfg: dict[str, Any], data_root: Path, ann_file_arg: str | None) -> Path:
    if ann_file_arg:
        return Path(ann_file_arg)

    val_dataset = cfg.get("val_dataloader", {}).get("dataset", {})
    ann_file = val_dataset.get("ann_file")
    if not ann_file:
        raise ValueError("Cannot resolve ann_file from config val_dataloader.dataset.ann_file.")

    ann_path = Path(str(ann_file))
    if ann_path.is_absolute():
        return ann_path
    return data_root / ann_path.name if ann_path.parent == Path(".") else data_root / ann_path


def resolve_checkpoint(cfg: dict[str, Any], checkpoint_arg: str | None) -> str:
    if checkpoint_arg:
        return checkpoint_arg
    checkpoint = cfg.get("load_from")
    if not checkpoint:
        raise ValueError("No checkpoint provided and config.load_from is empty.")
    return str(checkpoint)


def resolve_kitti_sample_paths(data_root: Path, img_rel: str, lidar_rel: str) -> tuple[Path, Path]:
    img_name = Path(img_rel).name
    lidar_name = Path(lidar_rel).name

    img_candidates = [
        data_root / img_rel,
        data_root / "training" / "image_2" / img_name,
        data_root / "testing" / "image_2" / img_name,
    ]
    lidar_candidates = [
        data_root / lidar_rel,
        data_root / "training" / "velodyne" / lidar_name,
        data_root / "testing" / "velodyne" / lidar_name,
    ]

    img_path = next((p for p in img_candidates if p.exists()), img_candidates[0])
    lidar_path = next((p for p in lidar_candidates if p.exists()), lidar_candidates[0])
    return img_path, lidar_path


def read_lidar_points(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Invalid lidar shape in {path}")
    return points.reshape(-1, 4)


def attack_lidar(points: np.ndarray, noise_std: float, drop_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = points.copy()
    out[:, :3] += rng.normal(0.0, noise_std, size=out[:, :3].shape).astype(np.float32)
    keep = rng.random(out.shape[0]) >= drop_ratio
    out = out[keep]
    if out.shape[0] == 0:
        return points[:1].copy()
    return out


def defend_lidar(points: np.ndarray, voxel_size: float) -> np.ndarray:
    o3d = try_import_open3d()
    if o3d is None:
        step = max(2, int(round(1.0 / max(voxel_size, 1e-6))))
        out = points[::step]
        return out if out.shape[0] > 0 else points[:1].copy()

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    downsampled = cloud.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(downsampled.points, dtype=np.float32)
    if xyz.shape[0] == 0:
        return points[:1].copy()
    intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate([xyz, intensity], axis=1)


def read_image(path: Path) -> np.ndarray:
    cv2 = try_import_cv2()
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        return img

    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1]


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2 = try_import_cv2()
    if cv2 is not None:
        cv2.imwrite(str(path), image_bgr)
        return

    from PIL import Image

    Image.fromarray(image_bgr[:, :, ::-1]).save(path)


def attack_image(image_bgr: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=image_bgr.shape).astype(np.float32)
    out = np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def defend_image(image_bgr: np.ndarray, ksize: int) -> np.ndarray:
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1

    cv2 = try_import_cv2()
    if cv2 is None:
        return image_bgr
    return cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def summarize_prediction(sample_id: str, pred: Any, score_thr: float, class_names: list[str], gt_count: int) -> dict[str, Any]:
    inst = pred.pred_instances_3d
    scores = to_numpy(inst.scores_3d)
    labels = to_numpy(inst.labels_3d).astype(np.int64)

    num_all = int(scores.shape[0])
    keep = scores >= score_thr
    num_keep = int(keep.sum())
    mean_score = float(scores.mean()) if num_all > 0 else 0.0
    mean_score_keep = float(scores[keep].mean()) if num_keep > 0 else 0.0

    class_hist: dict[str, int] = {name: 0 for name in class_names}
    for label in labels[keep]:
        if 0 <= int(label) < len(class_names):
            class_hist[class_names[int(label)]] += 1

    recall_proxy = float(min(num_keep, gt_count) / max(gt_count, 1))

    return {
        "sample_id": sample_id,
        "num_pred_all": num_all,
        "num_pred_thr": num_keep,
        "mean_score_all": mean_score,
        "mean_score_thr": mean_score_keep,
        "class_hist_thr": class_hist,
        "gt_count": int(gt_count),
        "recall_proxy": recall_proxy,
    }


def aggregate_branch(name: str, records: list[dict[str, Any]], class_names: list[str]) -> dict[str, Any]:
    num_samples = len(records)
    class_totals = {c: 0 for c in class_names}
    for r in records:
        for c in class_names:
            class_totals[c] += int(r["class_hist_thr"].get(c, 0))

    return {
        "branch": name,
        "num_samples": num_samples,
        "avg_num_pred_all": float(np.mean([r["num_pred_all"] for r in records])) if num_samples else 0.0,
        "avg_num_pred_thr": float(np.mean([r["num_pred_thr"] for r in records])) if num_samples else 0.0,
        "avg_mean_score_all": float(np.mean([r["mean_score_all"] for r in records])) if num_samples else 0.0,
        "avg_mean_score_thr": float(np.mean([r["mean_score_thr"] for r in records])) if num_samples else 0.0,
        "avg_recall_proxy": float(np.mean([r["recall_proxy"] for r in records])) if num_samples else 0.0,
        "class_totals_thr": class_totals,
    }


def delta_from_clean(clean: dict[str, Any], other: dict[str, Any]) -> dict[str, float]:
    return {
        "delta_avg_num_pred_thr": float(other["avg_num_pred_thr"] - clean["avg_num_pred_thr"]),
        "delta_avg_mean_score_thr": float(other["avg_mean_score_thr"] - clean["avg_mean_score_thr"]),
        "delta_avg_recall_proxy": float(other["avg_recall_proxy"] - clean["avg_recall_proxy"]),
    }


def compute_official_kitti_metrics(
    branch_results: list[Any],
    sample_indices: list[int],
    ann_file: Path,
    class_names: list[str],
) -> dict[str, float]:
    import os
    import shutil
    import sys
    import mmengine
    import tempfile

    from mmengine.structures import InstanceData
    from mmdet3d.evaluation.metrics import KittiMetric

    ann_infos = mmengine.load(str(ann_file))
    subset_infos = {
        "metainfo": ann_infos["metainfo"],
        "data_list": ann_infos["data_list"][:len(branch_results)],
    }

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvvm_candidates = [
        Path(sys.prefix)
        / "lib"
        / py_ver
        / "site-packages"
        / "nvidia"
        / "cuda_nvcc"
        / "nvvm"
        / "lib64"
        / "libnvvm.so",
        Path(sys.prefix) / "lib" / py_ver / "site-packages" / "nvidia" / "cu13" / "lib" / "libnvvm.so.4",
    ]
    for path_entry in sys.path:
        nvvm_candidates.append(Path(path_entry) / "nvidia" / "cu13" / "lib" / "libnvvm.so.4")

    for nvvm_so in nvvm_candidates:
        if not nvvm_so.exists():
            continue

        nvvm_dir = nvvm_so.parent
        symlink_path = nvvm_dir / "libnvvm.so"
        if not symlink_path.exists():
            try:
                symlink_path.symlink_to(nvvm_so.name)
            except OSError:
                shutil.copy2(nvvm_so, symlink_path)

        if nvvm_dir.name == "lib64" and nvvm_dir.parent.name == "nvvm":
            cuda_home = nvvm_dir.parent.parent
        else:
            cuda_home = nvvm_dir.parent
        compat_nvvm_dir = cuda_home / "nvvm" / "lib64"
        compat_nvvm_dir.mkdir(parents=True, exist_ok=True)
        for name in ["libnvvm.so", "libnvvm.so.4"]:
            compat_path = compat_nvvm_dir / name
            if compat_path.exists():
                continue
            try:
                compat_path.symlink_to(nvvm_so)
            except OSError:
                shutil.copy2(nvvm_so, compat_path)

        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ.setdefault("NUMBA_CUDA_DEFAULT_PTX_CC", "8.6")
        os.environ.setdefault("NUMBA_FORCE_CUDA_CC", "8.6")
        try:
            import numba.cuda.cudadrv.libs as numba_libs
            import numba.cuda.cuda_paths as numba_cuda_paths
            import numba.core.config as numba_config

            if hasattr(numba_libs.get_cuda_paths, "_cached_result"):
                delattr(numba_libs.get_cuda_paths, "_cached_result")
            if hasattr(numba_cuda_paths.get_cuda_paths, "_cached_result"):
                delattr(numba_cuda_paths.get_cuda_paths, "_cached_result")
            numba_config.FORCE_CUDA_CC = (8, 6)
            numba_config.CUDA_DEFAULT_PTX_CC = (8, 6)
        except Exception:
            pass
        break

    with tempfile.TemporaryDirectory() as tmp_dir:
        subset_ann_file = Path(tmp_dir) / "subset_kitti_infos.pkl"
        mmengine.dump(subset_infos, str(subset_ann_file))

        metric = KittiMetric(ann_file=str(subset_ann_file), metric="bbox")
        metric.dataset_meta = {"classes": tuple(class_names)}
        metric.data_infos = metric.convert_annos_to_kitti_annos(mmengine.load(str(subset_ann_file)))

        data_samples: list[dict[str, Any]] = []
        for pred, sample_idx in zip(branch_results, sample_indices):
            pred_3d_raw = pred.pred_instances_3d
            pred_3d = InstanceData()
            for key in pred_3d_raw.keys():
                value = pred_3d_raw[key]
                pred_3d[key] = value.to("cpu") if hasattr(value, "to") else value

            pred_2d_raw = pred.pred_instances if hasattr(pred, "pred_instances") else InstanceData()
            pred_2d = InstanceData()
            for key in pred_2d_raw.keys():
                value = pred_2d_raw[key]
                pred_2d[key] = value.to("cpu") if hasattr(value, "to") else value

            data_samples.append(
                dict(
                    pred_instances_3d=pred_3d,
                    pred_instances=pred_2d,
                    sample_idx=int(sample_idx),
                )
            )

        try:
            return metric.compute_metrics(data_samples)
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}; branch_results={len(branch_results)}; "
                f"sample_indices={sample_indices}; metric_results={len(data_samples)}; "
                f"data_infos={len(metric.data_infos)}"
            ) from exc


def main() -> int:
    args = parse_args()

    if args.check_env:
        ok, report = check_runtime_dependencies()
        print("[env] runtime dependency check")
        for name, value in report.items():
            print(f"  - {name}: {value}")
        return 0 if ok else 1

    mmengine = import_mmengine()

    cfg = load_config(Path(args.config))
    data_root = resolve_data_root(cfg, args.data_root)
    ann_file = resolve_ann_file(cfg, data_root, args.ann_file)
    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    class_names = list(cfg.get("class_names", ["Pedestrian", "Cyclist", "Car"]))

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    info = mmengine.load(str(ann_file))
    data_list = info["data_list"]
    selected = data_list[: args.num_samples]
    if len(selected) == 0:
        raise RuntimeError("No samples found in annotation file.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_adv_lidar = output_dir / "generated" / "adversarial" / "velodyne"
    tmp_adv_img = output_dir / "generated" / "adversarial" / "image_2"
    tmp_def_lidar = output_dir / "generated" / "defended" / "velodyne"
    tmp_def_img = output_dir / "generated" / "defended" / "image_2"
    for p in [tmp_adv_lidar, tmp_adv_img, tmp_def_lidar, tmp_def_img]:
        p.mkdir(parents=True, exist_ok=True)

    from mmdet3d.apis import inference_multi_modality_detector, init_model

    model = init_model(str(args.config), checkpoint, device=args.device)

    clean_pcds: list[str] = []
    clean_imgs: list[str] = []
    adv_pcds: list[str] = []
    adv_imgs: list[str] = []
    def_pcds: list[str] = []
    def_imgs: list[str] = []
    sample_ids: list[str] = []
    sample_indices: list[int] = []
    gt_counts: list[int] = []

    for idx, data_info in enumerate(selected):
        sample_indices.append(idx)
        img_rel = data_info["images"][args.cam_type]["img_path"]
        lidar_rel = data_info["lidar_points"]["lidar_path"]

        img_path, lidar_path = resolve_kitti_sample_paths(data_root, img_rel, lidar_rel)
        if not img_path.exists() or not lidar_path.exists():
            raise FileNotFoundError(f"Missing sample files: {img_path}, {lidar_path}")

        sample_id = Path(lidar_rel).stem
        sample_ids.append(sample_id)
        gt_counts.append(len(data_info.get("instances", [])))

        clean_pcds.append(str(lidar_path))
        clean_imgs.append(str(img_path))

        lidar_clean = read_lidar_points(lidar_path)
        img_clean = read_image(img_path)

        lidar_adv = attack_lidar(lidar_clean, args.noise_std, args.drop_ratio, seed=1000 + idx)
        img_adv = attack_image(img_clean, args.img_noise_std, seed=2000 + idx)
        lidar_def = defend_lidar(lidar_adv, args.voxel_size)
        img_def = defend_image(img_adv, args.img_blur_ksize)

        adv_lidar_path = tmp_adv_lidar / f"{sample_id}.bin"
        def_lidar_path = tmp_def_lidar / f"{sample_id}.bin"
        adv_img_path = tmp_adv_img / f"{sample_id}.png"
        def_img_path = tmp_def_img / f"{sample_id}.png"

        lidar_adv.astype(np.float32).tofile(adv_lidar_path)
        lidar_def.astype(np.float32).tofile(def_lidar_path)
        write_image(adv_img_path, img_adv)
        write_image(def_img_path, img_def)

        adv_pcds.append(str(adv_lidar_path))
        def_pcds.append(str(def_lidar_path))
        adv_imgs.append(str(adv_img_path))
        def_imgs.append(str(def_img_path))

    ann_metainfo = info.get("metainfo", {})

    def run_branch_inference(pcds: list[str], imgs: list[str], data_infos: list[dict[str, Any]], branch_name: str) -> list[Any]:
        branch_results: list[Any] = []
        branch_info_dir = output_dir / "generated" / branch_name / "infos"
        branch_info_dir.mkdir(parents=True, exist_ok=True)

        for sample_idx, (pcd_path, img_path, data_info) in enumerate(zip(pcds, imgs, data_infos)):
            subset_ann_file = branch_info_dir / f"{sample_idx:06d}.pkl"
            mmengine.dump({"metainfo": ann_metainfo, "data_list": [data_info]}, str(subset_ann_file))
            results, _ = inference_multi_modality_detector(
                model,
                [pcd_path],
                [img_path],
                str(subset_ann_file),
                cam_type=args.cam_type,
            )
            branch_results.extend(results)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        return branch_results

    clean_results = run_branch_inference(clean_pcds, clean_imgs, selected, "clean")
    adv_results = run_branch_inference(adv_pcds, adv_imgs, selected, "adversarial")
    def_results = run_branch_inference(def_pcds, def_imgs, selected, "defended")

    clean_records = [
        summarize_prediction(sample_ids[i], clean_results[i], args.score_thr, class_names, gt_counts[i])
        for i in range(len(sample_ids))
    ]
    adv_records = [
        summarize_prediction(sample_ids[i], adv_results[i], args.score_thr, class_names, gt_counts[i])
        for i in range(len(sample_ids))
    ]
    def_records = [
        summarize_prediction(sample_ids[i], def_results[i], args.score_thr, class_names, gt_counts[i])
        for i in range(len(sample_ids))
    ]

    clean_summary = aggregate_branch("clean", clean_records, class_names)
    adv_summary = aggregate_branch("adversarial", adv_records, class_names)
    def_summary = aggregate_branch("defended", def_records, class_names)

    official_metrics: dict[str, Any] = {}
    official_metric_error: str | None = None
    if not args.disable_official_kitti_metric:
        try:
            official_metrics = {
                "clean": compute_official_kitti_metrics(clean_results, sample_indices, ann_file, class_names),
                "adversarial": compute_official_kitti_metrics(adv_results, sample_indices, ann_file, class_names),
                "defended": compute_official_kitti_metrics(def_results, sample_indices, ann_file, class_names),
            }
        except Exception as exc:
            official_metric_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    report = {
        "config": args.config,
        "checkpoint": checkpoint,
        "ann_file": str(ann_file),
        "data_root": str(data_root),
        "device": args.device,
        "score_thr": args.score_thr,
        "attack": {
            "noise_std": args.noise_std,
            "drop_ratio": args.drop_ratio,
            "img_noise_std": args.img_noise_std,
        },
        "defense": {
            "voxel_size": args.voxel_size,
            "img_blur_ksize": args.img_blur_ksize,
        },
        "branches": {
            "clean": clean_summary,
            "adversarial": adv_summary,
            "defended": def_summary,
        },
        "deltas_vs_clean": {
            "adversarial": delta_from_clean(clean_summary, adv_summary),
            "defended": delta_from_clean(clean_summary, def_summary),
        },
        "per_sample": {
            "clean": clean_records,
            "adversarial": adv_records,
            "defended": def_records,
        },
        "official_kitti_metric": {
            "enabled": not args.disable_official_kitti_metric,
            "error": official_metric_error,
            "results": official_metrics,
        },
    }

    report_path = output_dir / "robustness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[done] multimodal robustness evaluation finished")
    print(f"[done] report: {report_path}")
    print(f"[done] clean avg pred@thr: {clean_summary['avg_num_pred_thr']:.4f}")
    print(f"[done] adv   avg pred@thr: {adv_summary['avg_num_pred_thr']:.4f}")
    print(f"[done] def   avg pred@thr: {def_summary['avg_num_pred_thr']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
