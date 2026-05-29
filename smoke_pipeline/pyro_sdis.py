"""Utilities for the PyroNear Pyro-SDIS Hugging Face dataset."""
from __future__ import annotations


def parse_yolo_annotation_string(raw: str | None) -> list[tuple[int, float, float, float, float]]:
    """
    Parse HF ``annotations`` field into (class, xc, yc, w, h) tuples.

    Multiple boxes are space-separated on one line, e.g.
    ``"1 0.1 0.2 0.03 0.04 1 0.5 0.6 0.02 0.03"``.
    """
    if not raw or not str(raw).strip():
        return []
    parts = str(raw).strip().split()
    boxes: list[tuple[int, float, float, float, float]] = []
    i = 0
    while i + 4 < len(parts):
        try:
            cls = int(float(parts[i]))
            xc, yc, bw, bh = map(float, parts[i + 1 : i + 5])
            boxes.append((cls, xc, yc, bw, bh))
            i += 5
        except (ValueError, IndexError):
            break
    return boxes


def normalize_label_lines(raw: str | None, smoke_class: int = 0) -> str:
    """Remap all boxes to a single smoke class id for YOLO training."""
    boxes = parse_yolo_annotation_string(raw)
    if not boxes:
        return ""
    lines = []
    for _cls, xc, yc, bw, bh in boxes:
        lines.append(f"{smoke_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + "\n"
