from __future__ import annotations

import cv2
import numpy as np


def estimate_horizon_row(
    bgr: np.ndarray,
    search_frac: float = 0.72,
    min_frac: float = 0.12,
    max_frac: float = 0.68,
) -> int:
    """
    Cheap skyline / horizon proxy for UAV nadir or oblique shots.

    Assumes sky occupies the upper portion; searches for a row with strong
    horizontal structure (common at tree/sky or mountain/sky boundaries).
    Falls back to a fixed band if the signal is weak.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    row_score = np.mean(np.abs(gx), axis=1)
    search = max(8, int(search_frac * h))
    band = row_score[:search]
    y = int(np.argmax(band))
    y_lo = int(min_frac * h)
    y_hi = int(max_frac * h)
    if y < y_lo or y > y_hi or float(band.max()) < 1e-3:
        y = int(0.42 * h)
    return int(np.clip(y, y_lo, y_hi))
