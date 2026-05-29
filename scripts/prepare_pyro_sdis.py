#!/usr/bin/env python3
"""
Download pyronear/pyro-sdis from Hugging Face and write Ultralytics YOLO layout.

Pyro-SDIS: fixed wildfire-camera frames (~1280px) with YOLO smoke boxes.
Official export: https://huggingface.co/datasets/pyronear/pyro-sdis

Example (full export, ~3.3 GB):
  python scripts/prepare_pyro_sdis.py --out data/pyro_sdis_yolo

Dry run (20 train + 10 val):
  python scripts/prepare_pyro_sdis.py --out data/pyro_sdis_yolo_tiny --max-samples 20
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from tqdm import tqdm

from smoke_pipeline.pyro_sdis import normalize_label_lines

REPO_ID = "pyronear/pyro-sdis"


def write_dataset_yaml(out: Path, splits: list[str]) -> None:
    lines = [f"path: {out.resolve()}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    if "train" in splits and "val" not in splits:
        lines.append("val: images/train")
    lines.extend(["names:", "  0: smoke", ""])
    (out / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")


def export_split(
    hf_split,
    split: str,
    out: Path,
    max_samples: int | None,
    labeled_only: bool,
) -> int:
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for row in tqdm(hf_split, desc=f"export {split}"):
        if max_samples is not None and n >= max_samples:
            break

        name = row["image_name"]
        ann = row.get("annotations")
        label_text = normalize_label_lines(ann, smoke_class=0)
        if labeled_only and not label_text.strip():
            continue

        image = row["image"]
        out_im = img_dir / name
        if not out_im.suffix:
            out_im = out_im.with_suffix(".jpg")
        image.save(str(out_im))

        out_lb = lbl_dir / f"{out_im.stem}.txt"
        out_lb.write_text(label_text, encoding="utf-8")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Pyro-SDIS from Hugging Face to YOLO layout.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/pyro_sdis_yolo"),
        help="Output root (images/<split>/, labels/<split>/, dataset.yaml)",
    )
    ap.add_argument(
        "--repo",
        type=str,
        default=REPO_ID,
        help="Hugging Face dataset id (default: pyronear/pyro-sdis)",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap rows per split (for quick tests)",
    )
    ap.add_argument(
        "--labeled-only",
        action="store_true",
        help="Skip images with no smoke boxes (~5k negatives dropped)",
    )
    ap.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val"],
        help="HF splits to export (default: train val)",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: pip install datasets huggingface_hub\n"
            f"  ({e})"
        ) from e

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.repo} from Hugging Face …")
    dataset = load_dataset(args.repo)

    written: list[str] = []
    for split in args.splits:
        if split not in dataset:
            print(f"skip {split}: not in dataset keys {list(dataset.keys())}")
            continue
        count = export_split(
            dataset[split],
            split,
            out,
            args.max_samples,
            args.labeled_only,
        )
        print(f"{split}: wrote {count} image/label pairs")
        if count:
            written.append(split)

    if not written:
        raise SystemExit("No splits exported.")

    write_dataset_yaml(out, written)
    print(f"wrote {out / 'dataset.yaml'}")
    print(
        "\nNext: build composite crops (sky-band + global panel), then train:\n"
        f"  python scripts/build_composite_dataset.py \\\n"
        f"    --in-root {out} \\\n"
        f"    --out-root {out.parent / (out.name + '_composite')} \\\n"
        f"    --size 640\n"
        f"  python scripts/train_yolo.py \\\n"
        f"    --data {out.parent / (out.name + '_composite')}/dataset.yaml \\\n"
        f"    --model yolo11s.pt --epochs 50 --name pyro_sdis_composite"
    )


if __name__ == "__main__":
    main()
