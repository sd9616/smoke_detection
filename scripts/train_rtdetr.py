#!/usr/bin/env python3
"""
Fine-tune Ultralytics RT-DETR on the same YOLO dataset.yaml as train_yolo.py.

Use the identical prepared data for a fair comparison with YOLO11 on FASDD_UAV, e.g.:
  data/fasdd_smoke_composite/dataset.yaml   (composite tiles, matches smoke_full_100ep)
  data/fasdd_smoke_half_composite/dataset.yaml
  data/fasdd_smoke_yolo/dataset.yaml        (raw frames, no composite crop)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from ultralytics import RTDETR


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RT-DETR on a smoke_detection dataset.yaml")
    ap.add_argument("--data", type=Path, required=True, help="dataset.yaml (same as YOLO11 runs)")
    ap.add_argument("--model", type=str, default="rtdetr-l.pt", help="rtdetr-l.pt, rtdetr-x.pt, …")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="RT-DETR often needs a smaller batch than YOLO")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Results root (default: runs/detect). Do not pass runs/detect — that double-nests paths.",
    )
    ap.add_argument("--name", type=str, default="rtdetr_fasdd")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument(
        "--val-conf",
        type=float,
        default=0.01,
        help="Val/plot confidence for val_batch*_pred.jpg",
    )
    args = ap.parse_args()

    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        composite_hint = data_path.parent.name.replace("_composite", "")
        raise SystemExit(
            f"Missing {data_path}\n"
            "Prepare FASDD first, then build composites to match YOLO11:\n"
            f"  python scripts/prepare_uav_subset.py --fasdd-root data/FASDD_UAV "
            f"--out data/{composite_hint} --all-images --smoke-only\n"
            f"  python scripts/build_composite_dataset.py "
            f"--in-root data/{composite_hint} --out-root {data_path.parent} --size 640"
        )

    model = RTDETR(args.model)
    train_kw: dict = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        show_labels=True,
        show_conf=True,
        conf=args.val_conf,
    )
    if args.project is not None:
        train_kw["project"] = str(args.project)
    if args.device:
        train_kw["device"] = args.device
    model.train(**train_kw)


if __name__ == "__main__":
    main()
