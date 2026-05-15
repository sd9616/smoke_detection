from __future__ import annotations

from pathlib import Path


def yolo_label_dir(fasdd_root: Path, split: str) -> Path | None:
    """Return .../annotations/YOLO/<split> if it exists."""
    p = fasdd_root / "annotations" / "YOLO" / split
    return p if p.is_dir() else None


def images_dir(fasdd_root: Path, split: str) -> Path | None:
    p = fasdd_root / "images" / split
    return p if p.is_dir() else None


def list_splits(fasdd_root: Path) -> list[str]:
    img_root = fasdd_root / "images"
    if not img_root.is_dir():
        return []
    return sorted(d.name for d in img_root.iterdir() if d.is_dir())
