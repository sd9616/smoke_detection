from __future__ import annotations

from typing import Mapping

import cv2
import numpy as np

# BGR colors aligned with Ultralytics val plots (class 0 fire, 1 smoke).
_CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "fire": (79, 68, 255),
    "smoke": (235, 219, 11),
}


def resolve_class_name(names: Mapping[int, str] | list[str], cls_id: int) -> str:
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def class_color_bgr(class_name: str, cls_id: int) -> tuple[int, int, int]:
    return _CLASS_COLORS_BGR.get(class_name.lower(), (40, 199, 40) if cls_id else (199, 40, 40))


def draw_detection(
    bgr: np.ndarray,
    box: tuple[int, int, int, int] | np.ndarray,
    cls_id: int,
    conf: float,
    names: Mapping[int, str] | list[str],
    *,
    line_thickness: int = 2,
) -> None:
    """Draw a box with a human-readable label, e.g. ``smoke 0.87``."""
    x1, y1, x2, y2 = map(int, box)
    class_name = resolve_class_name(names, cls_id)
    color = class_color_bgr(class_name, cls_id)
    label = f"{class_name} {conf:.2f}"
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, line_thickness)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(th + baseline + 2, y1)
    cv2.rectangle(bgr, (x1, ty - th - baseline - 2), (x1 + tw + 2, ty), color, -1)
    cv2.putText(
        bgr,
        label,
        (x1 + 1, ty - baseline - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
