#!/usr/bin/env python3
"""
Run a composite-tiled detector and map boxes back to the original full frame.

Outputs (x1,y1,x2,y2) in original pixels with confidence; applies NMS across
restored coordinates. Use a low --conf threshold when optimizing for recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from smoke_pipeline.composite import build_composite, composite_xyxy_to_original, nms_numpy
from smoke_pipeline.viz import draw_detection, resolve_class_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Image file or directory")
    ap.add_argument("--out", type=Path, default=Path("runs/pred_composite"))
    ap.add_argument("--imgsz", type=int, default=640, help="Must match training composite size")
    ap.add_argument("--conf", type=float, default=0.12, help="Lower for higher recall")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--save-txt", action="store_true")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    class_names = model.names
    src = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path]
    if src.is_file():
        paths = [src]
    else:
        paths = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) + sorted(src.glob("*.jpeg"))

    for im_path in tqdm(paths):
        bgr = cv2.imread(str(im_path))
        if bgr is None:
            continue
        comp, meta = build_composite(bgr, out_size=args.imgsz)
        res = model.predict(source=comp, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            cv2.imwrite(str(out / im_path.name), bgr)
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)

        mapped = np.stack([composite_xyxy_to_original(b, meta) for b in xyxy], axis=0)
        keep = nms_numpy(mapped, scores, iou_thresh=args.iou)
        mapped = mapped[keep]
        scores = scores[keep]
        cls = cls[keep]

        txt_path = out / f"{im_path.stem}.txt"
        if args.save_txt and txt_path.exists():
            txt_path.unlink()

        for box, cf, c in zip(mapped, scores, cls):
            draw_detection(bgr, box, int(c), float(cf), class_names)
            if args.save_txt:
                cname = resolve_class_name(class_names, int(c))
                line = f"{cname} {cf:.5f} " + " ".join(f"{v:.2f}" for v in box)
                with txt_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

        cv2.imwrite(str(out / im_path.name), bgr)

    print(f"saved under {out}")


if __name__ == "__main__":
    main()
