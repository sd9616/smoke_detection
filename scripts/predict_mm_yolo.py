#!/usr/bin/env python3
"""Run a trained RGB-thermal checkpoint and draw boxes on the RGB frame."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from mm_yolo.engine import predict_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="RGB-thermal smoke inference")
    ap.add_argument("--weights", type=Path, required=True, help="runs/detect/mm_yolo_rgbt/weights/best.pt")
    ap.add_argument("--rgb", type=Path, required=True, help="Directory of RGB frames")
    ap.add_argument("--thermal", type=Path, required=True, help="Directory of thermal frames, same filenames")
    ap.add_argument("--out", type=Path, default=Path("runs/pred_mm_yolo"))
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()
    predict_dir(
        weights=args.weights,
        rgb_dir=args.rgb,
        thermal_dir=args.thermal,
        out=args.out,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )


if __name__ == "__main__":
    main()
