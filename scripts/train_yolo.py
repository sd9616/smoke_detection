#!/usr/bin/env python3
"""Fine-tune Ultralytics YOLO11 on a prepared dataset.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dataset.yaml")
    ap.add_argument("--model", type=str, default="yolo11s.pt", help="Ultralytics weight name or path")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Results root (default: Ultralytics runs/detect). Do not pass runs/detect — that double-nests paths.",
    )
    ap.add_argument("--name", type=str, default="smoke_uav")
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()

    model = YOLO(args.model)
    train_kw: dict = dict(
        data=str(args.data.expanduser().resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        mosaic=1.0,
        copy_paste=0.15,
        degrees=4.0,
        translate=0.08,
        scale=0.55,
        fliplr=0.5,
        exist_ok=True,
        show_labels=True,
        show_conf=True,
    )
    if args.project is not None:
        train_kw["project"] = str(args.project)
    if args.device:
        train_kw["device"] = args.device
    model.train(**train_kw)


if __name__ == "__main__":
    main()
