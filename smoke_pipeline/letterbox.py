from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    im: np.ndarray,
    new_w: int,
    new_h: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize with unchanged aspect ratio, padding to (new_w, new_h).
    Returns (image_bgr, ratio, (dw, dh)) where ratio is scale applied to original.
    """
    h, w = im.shape[:2]
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = im if (nw, nh) == (w, h) else cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_w - nw) / 2, (new_h - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = np.full((new_h, new_w, 3), color, dtype=im.dtype)
    out[top : top + nh, left : left + nw] = resized
    return out, r, (float(left), float(top))


def xyxy_orig_to_letterbox(
    xyxy: np.ndarray,
    r: float,
    pad_xy: tuple[float, float],
) -> np.ndarray:
    """Map pixel xyxy from original image into letterboxed canvas."""
    left, top = pad_xy
    out = xyxy.copy().astype(np.float64)
    out[[0, 2]] = out[[0, 2]] * r + left
    out[[1, 3]] = out[[1, 3]] * r + top
    return out


def xyxy_letterbox_to_orig(
    xyxy: np.ndarray,
    r: float,
    pad_xy: tuple[float, float],
) -> np.ndarray:
    """Inverse of xyxy_orig_to_letterbox."""
    left, top = pad_xy
    out = xyxy.copy().astype(np.float64)
    out[[0, 2]] = (out[[0, 2]] - left) / r
    out[[1, 3]] = (out[[1, 3]] - top) / r
    return out


def clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x = xyxy.copy()
    x[0] = np.clip(x[0], 0, w - 1)
    x[1] = np.clip(x[1], 0, h - 1)
    x[2] = np.clip(x[2], 0, w - 1)
    x[3] = np.clip(x[3], 0, h - 1)
    return x
