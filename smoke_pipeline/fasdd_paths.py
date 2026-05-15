from __future__ import annotations

from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SPLIT_NAMES = ("train", "val", "test")
_YOLO_DIR_NAMES = ("YOLO_UAV", "YOLO")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTS


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if is_image_file(p))


def _split_subdirs(img_root: Path) -> list[Path]:
    return sorted(d for d in img_root.iterdir() if d.is_dir())


def yolo_annotation_root(fasdd_root: Path) -> Path | None:
    """Prefer annotations/YOLO_UAV, then annotations/YOLO."""
    ann = fasdd_root / "annotations"
    for name in _YOLO_DIR_NAMES:
        p = ann / name
        if p.is_dir():
            return p
    return None


def is_yolo_uav_list_layout(fasdd_root: Path) -> bool:
    """YOLO_UAV style: labels/ plus train.txt, val.txt, test.txt."""
    yroot = yolo_annotation_root(fasdd_root)
    if yroot is None or yroot.name != "YOLO_UAV":
        return False
    if not (yroot / "labels").is_dir():
        return False
    return any((yroot / f"{s}.txt").is_file() for s in _SPLIT_NAMES)


def is_flat_image_layout(fasdd_root: Path) -> bool:
    """True when images/ holds files directly (no train/val/test subfolders)."""
    img_root = fasdd_root / "images"
    if not img_root.is_dir():
        return False
    if _split_subdirs(img_root):
        return False
    return bool(list_images(img_root))


def _resolve_image_path(line: str, fasdd_root: Path, img_root: Path) -> Path | None:
    raw = line.strip().replace("\\", "/")
    if not raw or raw.startswith("#"):
        return None
    p = Path(raw)
    if p.is_absolute():
        return p.resolve() if is_image_file(p) else None

    rel = raw.lstrip("./")
    candidates = [
        fasdd_root / rel,
        img_root / rel,
        img_root / p.name,
        fasdd_root / "images" / rel,
        fasdd_root / "images" / p.name,
    ]
    for c in candidates:
        c = c.resolve()
        if is_image_file(c):
            return c
    return None


def _pairs_from_split_list(
    fasdd_root: Path, split: str, yroot: Path, img_root: Path
) -> list[tuple[Path, Path]]:
    list_file = yroot / f"{split}.txt"
    if not list_file.is_file():
        return []
    label_dir = yroot / "labels"
    pairs: list[tuple[Path, Path]] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        im = _resolve_image_path(line, fasdd_root, img_root)
        if im is None:
            continue
        lab = label_dir / f"{im.stem}.txt"
        if lab.is_file():
            pairs.append((im.resolve(), lab.resolve()))
    return pairs


def _pairs_from_image_dir(idir: Path, ldir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for im in list_images(idir):
        lab = ldir / f"{im.stem}.txt"
        if lab.is_file():
            pairs.append((im.resolve(), lab.resolve()))
    return pairs


def list_splits(fasdd_root: Path) -> list[str]:
    return [name for name, _, _ in resolve_splits(fasdd_root)]


def resolve_splits(fasdd_root: Path) -> list[tuple[str, Path, Path]]:
    """
    Return (split_name, image_dir, label_dir) for directory-scanned layouts.

  For YOLO_UAV list-file layout, use collect_split_pairs() instead; image_dir
  is still images/ (flat) and label_dir is annotations/YOLO_UAV/labels/.
    """
    fasdd_root = fasdd_root.resolve()
    img_root = fasdd_root / "images"
    if not img_root.is_dir():
        return []

    yroot = yolo_annotation_root(fasdd_root)
    if yroot is not None and is_yolo_uav_list_layout(fasdd_root):
        label_dir = yroot / "labels"
        return [
            (s, img_root, label_dir)
            for s in _SPLIT_NAMES
            if (yroot / f"{s}.txt").is_file()
        ]

    split_dirs = _split_subdirs(img_root)
    if split_dirs and yroot is not None:
        out: list[tuple[str, Path, Path]] = []
        for d in split_dirs:
            ldir = yroot / d.name
            if ldir.is_dir():
                out.append((d.name, d, ldir))
        if out:
            return out

    if yroot is not None:
        split_dirs = [yroot / s for s in _SPLIT_NAMES if (yroot / s).is_dir()]
        if split_dirs:
            return [(d.name, img_root, d) for d in split_dirs]

        if is_flat_image_layout(fasdd_root) and (yroot / "labels").is_dir():
            return [("train", img_root, yroot / "labels")]
        if is_flat_image_layout(fasdd_root):
            return [("train", img_root, yroot)]

    return []


def collect_split_pairs(fasdd_root: Path, split: str) -> list[tuple[Path, Path]]:
    """Image/label path pairs for one split (list-file or directory layout)."""
    fasdd_root = fasdd_root.resolve()
    img_root = fasdd_root / "images"
    yroot = yolo_annotation_root(fasdd_root)

    if yroot is not None and is_yolo_uav_list_layout(fasdd_root):
        return _pairs_from_split_list(fasdd_root, split, yroot, img_root)

    for name, idir, ldir in resolve_splits(fasdd_root):
        if name == split:
            return _pairs_from_image_dir(idir, ldir)
    return []


def yolo_label_dir(fasdd_root: Path, split: str) -> Path | None:
    yroot = yolo_annotation_root(fasdd_root)
    if yroot is None:
        return None
    if is_yolo_uav_list_layout(fasdd_root):
        p = yroot / "labels"
        return p if p.is_dir() else None
    if is_flat_image_layout(fasdd_root) and (yroot / "labels").is_dir():
        return yroot / "labels"
    p = yroot / split
    return p if p.is_dir() else None


def images_dir(fasdd_root: Path, split: str) -> Path | None:
    img_root = fasdd_root / "images"
    if not img_root.is_dir():
        return None
    if is_flat_image_layout(fasdd_root) or is_yolo_uav_list_layout(fasdd_root):
        return img_root
    p = img_root / split
    return p if p.is_dir() else None
