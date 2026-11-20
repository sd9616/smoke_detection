#!/usr/bin/env python3
"""Run the attention model (RT-DETR) on images. Does not run YOLO."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from ultralytics import RTDETR

DEFAULT_WEIGHTS = (
    Path("runs/detect/attention_images/weights/best.pt"),
    Path("runs/detect/rtdetr_l_fasdd_full/weights/best.pt"),
)


def resolve_weights(path: Path | None) -> Path:
    if path is not None:
        candidate = path.expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise SystemExit(f"Missing attention weights: {candidate}")
    for candidate in DEFAULT_WEIGHTS:
        if candidate.is_file():
            print(f"Reusing attention checkpoint: {candidate.resolve()}")
            return candidate.resolve()
    raise SystemExit(
        "No attention checkpoint found.\n"
        "Train it on the images with scripts/train_attention.py, "
        "or pass --weights runs/detect/rtdetr_l_fasdd_full/weights/best.pt"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="RT-DETR inference on an image file or directory")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--source", type=Path, required=True, help="Image file or directory (the images, not a second composite pass)")
    ap.add_argument("--out", type=Path, default=Path("runs/pred_attention"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    source = args.source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Missing source: {source}")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    model = RTDETR(str(weights))
    predict_kw: dict = dict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        save=True,
        show_labels=True,
        show_conf=True,
    )
    if args.device:
        predict_kw["device"] = args.device
    model.predict(**predict_kw)
    print(f"saved under {out}")


if __name__ == "__main__":
    main()
