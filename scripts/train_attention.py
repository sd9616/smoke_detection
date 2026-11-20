#!/usr/bin/env python3
"""Train the attention model (RT-DETR) on the prepared images.

YOLO is a separate model, trained on the raw frames. This script never starts
that run. If runs/detect/smoke_full_100ep/weights/best.pt is already on disk,
it is left untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from ultralytics import RTDETR

YOLO_WEIGHTS = Path("runs/detect/smoke_full_100ep/weights/best.pt")
ATTENTION_WEIGHTS = (
    Path("runs/detect/attention_images/weights/best.pt"),
    Path("runs/detect/rtdetr_l_fasdd_full/weights/best.pt"),
)
DEFAULT_IMAGES = Path("data/fasdd_smoke_composite/dataset.yaml")


def existing_attention(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.expanduser().is_file():
        return explicit.expanduser().resolve()
    for path in ATTENTION_WEIGHTS:
        if path.is_file():
            return path.resolve()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RT-DETR on images. Does not train YOLO.")
    ap.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Image dataset.yaml (composite tiles, not the raw-frame YOLO set)",
    )
    ap.add_argument("--model", type=str, default="rtdetr-l.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--name", type=str, default="attention_images")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--val-conf", type=float, default=0.15)
    ap.add_argument(
        "--attention-weights",
        type=Path,
        default=None,
        help="Existing attention checkpoint to reuse instead of training",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Train again even when an attention checkpoint is already on disk",
    )
    args = ap.parse_args()

    if YOLO_WEIGHTS.is_file():
        print(f"YOLO checkpoint already present: {YOLO_WEIGHTS.resolve()}")
        print("Not training YOLO.")
    else:
        print("No finished YOLO checkpoint at runs/detect/smoke_full_100ep/weights/best.pt.")
        print("This script does not train it. On the raw frames, when you need that model:")
        print(
            "  python scripts/train_yolo.py "
            "--data data/fasdd_smoke_yolo/dataset.yaml --model yolo11s.pt "
            "--epochs 100 --name smoke_full_100ep --device 0"
        )

    found = existing_attention(args.attention_weights)
    if found is not None and not args.force:
        print(f"Attention model already trained: {found}")
        print("Not repeating it. Pass --force to train again.")
        return

    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(
            f"Missing image dataset {data_path}\n"
            "Build the composite images first, then train only the attention model:\n"
            "  python scripts/build_composite_dataset.py "
            "--in-root data/fasdd_smoke_yolo "
            "--out-root data/fasdd_smoke_composite --size 640\n"
            "  python scripts/train_attention.py --data data/fasdd_smoke_composite/dataset.yaml --device 0"
        )

    print(f"Training the attention model on images: {data_path}")
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
