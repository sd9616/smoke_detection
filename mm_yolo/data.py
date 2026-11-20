"""Paired RGB / thermal frames with shared YOLO labels."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from smoke_pipeline.composite import build_composite, transform_labels_to_composite
from smoke_pipeline.horizon import estimate_horizon_row

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_dataset_yaml(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Missing dataset yaml: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    root = Path(data["path"])
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    data["path"] = root
    data["_yaml"] = path
    return data


def split_dirs(root: Path, split: str) -> tuple[Path, Path, Path]:
    """Return rgb dir, thermal dir, and label dir for one split."""
    rgb_dir = root / split / "rgb"
    thermal_dir = root / split / "thermal"
    label_dir = root / "labels" / Path(split).name
    return rgb_dir, thermal_dir, label_dir


def list_pairs(root: Path, split: str) -> list[tuple[Path, Path, Path]]:
    rgb_dir, thermal_dir, label_dir = split_dirs(root, split)
    if not rgb_dir.is_dir() or not thermal_dir.is_dir():
        raise SystemExit(
            f"Missing paired images under {root / split}\n"
            "Expected:\n"
            f"  {rgb_dir}/\n"
            f"  {thermal_dir}/\n"
            f"  {label_dir}/   (YOLO txt, same stem as the image)"
        )
    pairs = []
    missing_thermal = []
    for rgb_path in sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        thermal_path = _match_stem(thermal_dir, rgb_path.stem)
        if thermal_path is None:
            missing_thermal.append(rgb_path.name)
            continue
        pairs.append((rgb_path, thermal_path, label_dir / f"{rgb_path.stem}.txt"))
    if missing_thermal:
        sample = ", ".join(missing_thermal[:5])
        raise SystemExit(
            f"{len(missing_thermal)} RGB frames in {rgb_dir} have no thermal image with the same stem "
            f"({sample})."
        )
    if not pairs:
        raise SystemExit(f"No images in {rgb_dir}")
    return pairs


def _match_stem(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def to_chw(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)


def read_labels(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RGBThermalDataset(Dataset):
    """Aligned RGB and thermal frames. Labels are in the original RGB frame."""

    def __init__(
        self,
        root: Path,
        split: str,
        imgsz: int = 640,
        composite: bool = True,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.composite = composite
        self.augment = augment
        self.pairs = list_pairs(self.root, split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        rgb_path, thermal_path, label_path = self.pairs[index]
        rgb = read_bgr(rgb_path)
        thermal = read_bgr(thermal_path)
        if thermal.shape[:2] != rgb.shape[:2]:
            thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        labels = read_labels(label_path)

        if self.composite:
            horizon = estimate_horizon_row(rgb)
            rgb, meta = build_composite(rgb, out_size=self.imgsz, horizon_y=horizon)
            thermal, _ = build_composite(thermal, out_size=self.imgsz, horizon_y=horizon)
            labels = transform_labels_to_composite(labels, meta)
        else:
            rgb = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            thermal = cv2.resize(thermal, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        cls, boxes = _parse_yolo(labels)
        if self.augment and np.random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            thermal = np.ascontiguousarray(thermal[:, ::-1])
            if len(boxes):
                boxes[:, 0] = 1.0 - boxes[:, 0]

        return {
            "rgb": to_chw(rgb),
            "thermal": to_chw(thermal),
            "cls": cls,
            "bboxes": boxes,
            "path": str(rgb_path),
        }


def _parse_yolo(lines: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    classes: list[float] = []
    boxes: list[list[float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        classes.append(float(parts[0]))
        boxes.append([float(v) for v in parts[1:]])
    if not boxes:
        return torch.zeros(0, 1), torch.zeros(0, 4)
    return torch.tensor(classes, dtype=torch.float32).view(-1, 1), torch.tensor(boxes, dtype=torch.float32)


def collate_pairs(batch: list[dict]) -> dict:
    classes = []
    boxes = []
    batch_idx = []
    for index, sample in enumerate(batch):
        count = sample["bboxes"].shape[0]
        if count == 0:
            continue
        classes.append(sample["cls"])
        boxes.append(sample["bboxes"])
        batch_idx.append(torch.full((count,), index, dtype=torch.float32))
    if boxes:
        cls = torch.cat(classes, 0)
        bboxes = torch.cat(boxes, 0)
        batch_index = torch.cat(batch_idx, 0)
    else:
        cls = torch.zeros(0, 1)
        bboxes = torch.zeros(0, 4)
        batch_index = torch.zeros(0)
    return {
        "rgb": torch.stack([sample["rgb"] for sample in batch], 0),
        "thermal": torch.stack([sample["thermal"] for sample in batch], 0),
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_index,
        "paths": [sample["path"] for sample in batch],
    }
