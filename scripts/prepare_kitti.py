#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KITTI folder and generate kitti_infos_*.pkl if needed."
    )
    parser.add_argument(
        "--data-root",
        default="data/kitti",
        help="KITTI dataset root directory.",
    )
    parser.add_argument(
        "--extra-tag",
        default="kitti",
        help="Prefix used by MMDetection3D create_data.py.",
    )
    parser.add_argument(
        "--with-plane",
        action="store_true",
        help="Whether to include road plane info while generating metadata.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only validate structure, do not generate infos.",
    )
    return parser.parse_args()


def required_paths(root: Path) -> list[Path]:
    return [
        root / "ImageSets" / "train.txt",
        root / "ImageSets" / "val.txt",
        root / "training" / "image_2",
        root / "training" / "velodyne",
        root / "training" / "calib",
        root / "training" / "label_2",
    ]


def info_files(root: Path, tag: str) -> list[Path]:
    return [
        root / f"{tag}_infos_train.pkl",
        root / f"{tag}_infos_val.pkl",
    ]


def find_create_data_script() -> Path | None:
    try:
        import mmdet3d  # type: ignore
    except Exception:
        return None

    mmdet3d_root = Path(mmdet3d.__file__).resolve().parent
    candidate = mmdet3d_root / ".mim" / "tools" / "create_data.py"
    if candidate.exists():
        return candidate
    return None


def validate_structure(root: Path) -> tuple[bool, list[Path]]:
    missing = [p for p in required_paths(root) if not p.exists()]
    return len(missing) == 0, missing


def run_create_data(create_script: Path, root: Path, tag: str, with_plane: bool) -> int:
    # mmdet3d create_data.py uses imports like `from tools...`, so execute it
    # as a module from the `.mim` root where `tools/` is importable.
    run_cwd = create_script.parent.parent
    cmd = [
        sys.executable,
        "-m",
        "tools.create_data",
        "kitti",
        "--root-path",
        str(root),
        "--out-dir",
        str(root),
        "--extra-tag",
        tag,
    ]
    if with_plane:
        cmd.append("--with-plane")

    print("[info] running:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False, cwd=run_cwd)
    return completed.returncode


def main() -> int:
    args = parse_args()
    root = Path(args.data_root).resolve()

    ok, missing = validate_structure(root)
    if not ok:
        print("[error] KITTI directory is incomplete. Missing:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print(f"[ok] KITTI structure looks valid: {root}")

    infos = info_files(root, args.extra_tag)
    if all(p.exists() for p in infos):
        print("[ok] metadata files already exist:")
        for path in infos:
            print(f"  - {path}")
        return 0

    if args.skip_generate:
        print("[warn] metadata files are missing and --skip-generate is set.")
        for path in infos:
            print(f"  - expected: {path}")
        return 1

    create_script = find_create_data_script()
    if create_script is None:
        print("[error] cannot find mmdet3d create_data.py.")
        print("[hint] ensure mmdet3d is installed in the active environment.")
        return 1

    print(f"[info] found create_data.py: {create_script}")
    code = run_create_data(create_script, root, args.extra_tag, args.with_plane)
    if code != 0:
        # Some mmdet3d versions may fail while post-processing test split info,
        # but train/val metadata can still be generated successfully.
        if all(p.exists() for p in infos):
            print(f"[warn] create_data.py exited with code {code}, but required train/val metadata exists.")
            print("[warn] continuing because training/evaluation only requires train/val info files.")
            return 0
        print(f"[error] metadata generation failed with exit code {code}")
        return code

    if all(p.exists() for p in infos):
        print("[ok] KITTI is ready. Generated:")
        for path in infos:
            print(f"  - {path}")
        return 0

    print("[error] create_data.py finished but metadata files are still missing.")
    for path in infos:
        print(f"  - expected: {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
