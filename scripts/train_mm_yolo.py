#!/usr/bin/env python3
"""Train RGB-thermal YOLO11. The RGB stream reuses the finished smoke checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from mm_yolo.engine import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune cross-attention on paired RGB and thermal frames")
    ap.add_argument("--data", type=Path, required=True, help="dataset.yaml with images/<split>/{rgb,thermal}")
    ap.add_argument(
        "--rgb-weights",
        type=Path,
        default=None,
        help="YOLO11s smoke checkpoint. Default: runs/detect/smoke_full_100ep/weights/best.pt when that file exists",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", type=Path, default=Path("runs/detect"))
    ap.add_argument("--name", type=str, default="mm_yolo_rgbt")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--unfreeze-rgb", action="store_true", help="Also train the RGB backbone")
    ap.add_argument("--unfreeze-head", action="store_true", help="Also train the neck and detection head")
    ap.add_argument(
        "--no-composite",
        action="store_true",
        help="Resize the raw pair to imgsz instead of the composite tile the smoke model was trained on",
    )
    args = ap.parse_args()
    train(
        data=args.data,
        rgb_weights=args.rgb_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        lr=args.lr,
        patience=args.patience,
        freeze_rgb=not args.unfreeze_rgb,
        freeze_head=not args.unfreeze_head,
        composite=not args.no_composite,
    )


if __name__ == "__main__":
    main()
