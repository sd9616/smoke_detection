#!/usr/bin/env python3
"""
Stage 1: build a lean YOLO layout from unpacked FASDD (Science Data Bank).

Expected FASDD_UAV layout (flat images + YOLO_UAV list files):

  <root>/images/*.jpg
  <root>/annotations/YOLO_UAV/labels/*.txt
  <root>/annotations/YOLO_UAV/{train,val,test}.txt

Also supported: images/{train,val,test}/ with annotations/YOLO/<split>/.

Science Data Bank also publishes a UAV-only archive (FASDD_UAV.zip). After
unpacking it, pass that folder as --fasdd-root and use --all-images (every
frame is already UAV).

For mixed FASDD_CV only, omit --all-images and use --uav-substrings to filter
paths by filename hints (uav, drone, dji, ...).

Example (UAV-only zip):
  .venv/bin/python scripts/prepare_uav_subset.py \\
    --fasdd-root /data/FASDD_UAV \\
    --out data/fasdd_uav_yolo \\
    --all-images

Example (0.1% random subset for a quick dry run):
  .venv/bin/python scripts/prepare_uav_subset.py \\
    --fasdd-root data/FASDD_UAV \\
    --out data/fasdd_uav_yolo_0p1pct \\
    --all-images --sample-frac 0.001 --sample-seed 42

Example (mixed CV, path filter):
  .venv/bin/python scripts/prepare_uav_subset.py \\
    --fasdd-root /data/FASDD_CV \\
    --out data/fasdd_uav_yolo \\
    --uav-substrings uav drone dji
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
import random
import shutil
from pathlib import Path

from smoke_pipeline.fasdd_paths import (
    collect_split_pairs,
    is_yolo_uav_list_layout,
    resolve_splits,
)


def should_keep(path: Path, substrings: list[str]) -> bool:
    s = str(path).lower()
    return any(t.lower() in s for t in substrings)


def symlink_relative(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src, start=dst.parent)
    os.symlink(rel, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--fasdd-root",
        type=Path,
        help="Unpacked FASDD_UAV or FASDD_CV root (see script docstring).",
    )
    src.add_argument(
        "--fasdd-cv",
        type=Path,
        help="Deprecated alias for --fasdd-root (kept for older commands).",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output dataset root (YOLO layout)")
    ap.add_argument(
        "--uav-substrings",
        nargs="*",
        default=["uav", "drone", "dj"],
        help="Keep images whose full path contains any token (case-insensitive).",
    )
    ap.add_argument("--all-images", action="store_true", help="Disable UAV path filtering")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        metavar="F",
        help="After filtering, keep a random F fraction of images with labels per split (e.g. 0.001 = 0.1%%).",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed when --sample-frac is set.",
    )
    args = ap.parse_args()

    root = (args.fasdd_root or args.fasdd_cv).expanduser().resolve()
    out = args.out.expanduser().resolve()
    partitions = resolve_splits(root)
    if not partitions:
        raise SystemExit(
            f"No dataset under {root}. "
            "Expected images/ plus annotations/YOLO_UAV/labels/ and train.txt (etc.), "
            "or images/<split>/ with annotations/YOLO/<split>/."
        )
    if is_yolo_uav_list_layout(root):
        print("using annotations/YOLO_UAV list files (train.txt, val.txt, test.txt)")

    written_splits: list[str] = []
    for split, _idir, _ldir in partitions:
        pairs = collect_split_pairs(root, split)
        if not args.all_images:
            pairs = [(im, lab) for im, lab in pairs if should_keep(im, args.uav_substrings)]
        if args.sample_frac is not None:
            f = args.sample_frac
            if not 0 < f <= 1:
                raise SystemExit("--sample-frac must be in (0, 1].")
            k = max(1, int(len(pairs) * f))
            k = min(k, len(pairs))
            rng = random.Random(args.sample_seed)
            pairs = rng.sample(pairs, k=k) if pairs else []
        o_img = out / "images" / split
        o_lbl = out / "labels" / split
        o_img.mkdir(parents=True, exist_ok=True)
        o_lbl.mkdir(parents=True, exist_ok=True)
        n = 0
        for im, lab in pairs:
            if args.copy:
                shutil.copy2(im, o_img / im.name)
                shutil.copy2(lab, o_lbl / f"{im.stem}.txt")
            else:
                symlink_relative(im, o_img / im.name)
                symlink_relative(lab, o_lbl / f"{im.stem}.txt")
            n += 1
        note = f" (sample-frac={args.sample_frac}, seed={args.sample_seed})" if args.sample_frac else ""
        print(f"{split}: linked/copied {n} image/label pairs{note}")
        if n:
            written_splits.append(split)

    yaml_path = out / "dataset.yaml"
    yaml_lines = [f"path: {out}"]
    for role in ("train", "val", "test"):
        if role in written_splits:
            yaml_lines.append(f"{role}: images/{role}")
    if "train" in written_splits and "val" not in written_splits:
        yaml_lines.append("val: images/train")
    yaml_lines.extend(["names:", "  0: fire", "  1: smoke", ""])
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"wrote {yaml_path} — verify class ids 0/1 match your label files.")


if __name__ == "__main__":
    main()
