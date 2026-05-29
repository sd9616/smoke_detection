#!/usr/bin/env python3
"""
Train YOLO11 on Pyro-SDIS composite crops (recommended defaults).

Assumes you already ran:
  1. scripts/prepare_pyro_sdis.py
  2. scripts/build_composite_dataset.py --in-root data/pyro_sdis_yolo ...

This is a thin wrapper around train_yolo.py with Pyro-SDIS-friendly defaults.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train YOLO11 on Pyro-SDIS composite dataset.")
    ap.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data/pyro_sdis_yolo_composite/dataset.yaml",
        help="Composite dataset.yaml",
    )
    ap.add_argument("--model", type=str, default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", type=str, default="pyro_sdis_composite")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--val-conf", type=float, default=0.12)
    args = ap.parse_args()

    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"Missing {data_path}\n"
            "Run:\n"
            "  python scripts/prepare_pyro_sdis.py --out data/pyro_sdis_yolo\n"
            "  python scripts/build_composite_dataset.py \\\n"
            "    --in-root data/pyro_sdis_yolo --out-root data/pyro_sdis_yolo_composite"
        )

    cmd = [
        sys.executable,
        str(_ROOT / "scripts/train_yolo.py"),
        "--data",
        str(data_path),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--name",
        args.name,
        "--val-conf",
        str(args.val_conf),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
