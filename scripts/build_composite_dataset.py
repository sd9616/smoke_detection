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
from smoke_pipeline.fasdd_paths import list_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--size", type=int, default=640, help="Square composite edge length")
    ap.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits under images/ (default: all subdirs found in --in-root)",
    )
    args = ap.parse_args()

    src = args.in_root.expanduser().resolve()
    dst = args.out_root.expanduser().resolve()
    img_root = src / "images"
    if args.splits:
        splits = args.splits
    elif img_root.is_dir():
        splits = sorted(d.name for d in img_root.iterdir() if d.is_dir())
    else:
        splits = []

    written_splits: list[str] = []
    for split in splits:
        idir = src / "images" / split
        ldir = src / "labels" / split
        if not idir.is_dir() or not ldir.is_dir():
            print(f"skip {split}: missing images or labels")
            continue
        o_img = dst / "images" / split
        o_lbl = dst / "labels" / split
        o_img.mkdir(parents=True, exist_ok=True)
        o_lbl.mkdir(parents=True, exist_ok=True)
        imgs = list_images(idir)
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
        written_splits.append(split)

    yaml = dst / "dataset.yaml"
    yaml_lines = [f"path: {dst}"]
    for role in ("train", "val", "test"):
        if role in written_splits:
            yaml_lines.append(f"{role}: images/{role}")
    if "train" in written_splits and "val" not in written_splits:
        yaml_lines.append("val: images/train")
    yaml_lines.extend(["names:", "  0: fire", "  1: smoke", ""])
    yaml.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"wrote composites under {dst}")


if __name__ == "__main__":
    main()
