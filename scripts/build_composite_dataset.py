#!/usr/bin/env python3
"""
Stage 2: rasterize Jung-style composite tiles + remap labels into composite space.

Reads a YOLO dataset produced by prepare_uav_subset.py and writes:
  <out>/images/<split>/*.jpg
  <out>/labels/<split>/*.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import cv2
from tqdm import tqdm

from smoke_pipeline.composite import build_composite, transform_labels_to_composite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--size", type=int, default=640, help="Square composite edge length")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    args = ap.parse_args()

    src = args.in_root.expanduser().resolve()
    dst = args.out_root.expanduser().resolve()

    for split in args.splits:
        idir = src / "images" / split
        ldir = src / "labels" / split
        if not idir.is_dir() or not ldir.is_dir():
            print(f"skip {split}: missing images or labels")
            continue
        o_img = dst / "images" / split
        o_lbl = dst / "labels" / split
        o_img.mkdir(parents=True, exist_ok=True)
        o_lbl.mkdir(parents=True, exist_ok=True)
        imgs = sorted(idir.glob("*.jpg")) + sorted(idir.glob("*.png")) + sorted(idir.glob("*.jpeg"))
        for im_path in tqdm(imgs, desc=split):
            lab_path = ldir / f"{im_path.stem}.txt"
            if not lab_path.is_file():
                continue
            bgr = cv2.imread(str(im_path))
            if bgr is None:
                continue
            comp, meta = build_composite(bgr, out_size=args.size)
            lines = lab_path.read_text(encoding="utf-8").splitlines()
            new_lines = transform_labels_to_composite(lines, meta)
            out_im = o_img / f"{im_path.stem}.jpg"
            out_lb = o_lbl / f"{im_path.stem}.txt"
            cv2.imwrite(str(out_im), comp)
            out_lb.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    yaml = dst / "dataset.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {dst}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: fire",
                "  1: smoke",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote composites under {dst}")


if __name__ == "__main__":
    main()
