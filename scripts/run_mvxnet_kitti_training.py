#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return runpy.run_path(str(config_path))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch training for the MVXNet-style KITTI multimodal config."
    )
    parser.add_argument(
        "--config",
        default="configs/multimodal/mvxnet_kitti_3class.py",
        help="Path to the training config.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Override the work directory from the config.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="Distributed launcher type used by MMEngine.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and write a manifest without starting training.",
    )
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="Directory to store the launch manifest. Defaults to the effective work dir.",
    )
    parser.add_argument(
        "--allow-system-python",
        action="store_true",
        help="Allow launching with a Python executable outside the project .venv.",
    )
    return parser.parse_known_args()


def expected_project_python() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if sys.platform.startswith("win"):
        return project_root / ".venv" / "Scripts" / "python.exe"
    return project_root / ".venv" / "bin" / "python"


def ensure_python_consistency(allow_system_python: bool) -> None:
    expected_python = expected_project_python().resolve()
    current_python = Path(sys.executable).resolve()

    if allow_system_python:
        return

    if expected_python.exists() and current_python != expected_python:
        raise RuntimeError(
            "Interpreter mismatch detected. "
            f"Current: {current_python}; Expected: {expected_python}. "
            "Activate the project .venv or pass --allow-system-python."
        )


def resolve_effective_work_dir(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)

    work_dir = cfg.get("work_dir")
    if work_dir:
        return Path(str(work_dir))

    experiment_name = str(cfg.get("experiment_name", "mvxnet_kitti_3class"))
    return Path("work_dirs") / experiment_name


def build_train_command(config_path: Path, work_dir: Path, launcher: str, extra_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "mim",
        "train",
        "mmdet3d",
        str(config_path),
        "--launcher",
        launcher,
        "--work-dir",
        str(work_dir),
    ]
    cmd.extend(extra_args)
    return cmd


def write_manifest(manifest_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main() -> int:
    args, extra_args = parse_args()
    ensure_python_consistency(args.allow_system_python)

    config_path = Path(args.config)
    cfg = load_config(config_path)

    effective_work_dir = resolve_effective_work_dir(cfg, args.work_dir)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else effective_work_dir
    command = build_train_command(config_path, effective_work_dir, args.launcher, extra_args)

    manifest = {
        "config": str(config_path),
        "experiment_name": cfg.get("experiment_name"),
        "work_dir": str(effective_work_dir),
        "python_executable": str(Path(sys.executable).resolve()),
        "launcher": args.launcher,
        "command": command,
        "extra_args": extra_args,
    }
    manifest_path = write_manifest(manifest_dir, manifest)

    print("[info] launch manifest written")
    print(f"[info] {manifest_path}")
    print("[info] training command")
    print(" ".join(command))

    if args.dry_run:
        print("[info] dry-run requested; training not started")
        return 0

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"[error] training command failed with exit code {completed.returncode}")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())