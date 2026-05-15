from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from smoke_pipeline.horizon import estimate_horizon_row
from smoke_pipeline.letterbox import clip_xyxy, letterbox, xyxy_letterbox_to_orig, xyxy_orig_to_letterbox


@dataclass
class CompositeMeta:
    """Metadata to map boxes between original image and composite square."""

    out_size: int
    panel_h: int
    orig_h: int
    orig_w: int
    sky_y2: int
    top_r: float
    top_pad: tuple[float, float]
    bot_r: float
    bot_pad: tuple[float, float]


def yolo_norm_to_xyxy(line: str, w: int, h: int) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    c = int(parts[0])
    xc, yc, bw, bh = map(float, parts[1:])
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return c, x1, y1, x2, y2


def xyxy_to_yolo_norm(c: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return f"{c} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}"


def build_composite(
    bgr: np.ndarray,
    out_size: int = 640,
    horizon_y: int | None = None,
    sky_margin_frac: float = 0.04,
) -> tuple[np.ndarray, CompositeMeta]:
    """
    Stack a global letterboxed view (top half) and a sky-focused letterboxed
    crop (bottom half) into one square — same spirit as Jung et al.'s
    skyline-guided multi-resolution input, without a second forward pass.
    """
    h, w = bgr.shape[:2]
    if horizon_y is None:
        horizon_y = estimate_horizon_row(bgr)
    margin = max(12, int(sky_margin_frac * h))
    sky_y2 = int(np.clip(horizon_y + margin, int(0.08 * h), h))

    panel_h = out_size // 2
    top_canvas, top_r, top_pad = letterbox(bgr, out_size, panel_h)

    sky = bgr[:sky_y2, :, :]
    bot_canvas, bot_r, bot_pad = letterbox(sky, out_size, panel_h)

    comp = np.zeros((out_size, out_size, 3), dtype=bgr.dtype)
    comp[:panel_h, :, :] = top_canvas
    comp[panel_h:, :, :] = bot_canvas

    meta = CompositeMeta(
        out_size=out_size,
        panel_h=panel_h,
        orig_h=h,
        orig_w=w,
        sky_y2=sky_y2,
        top_r=top_r,
        top_pad=top_pad,
        bot_r=bot_r,
        bot_pad=bot_pad,
    )
    return comp, meta


def transform_labels_to_composite(
    label_lines: Iterable[str],
    meta: CompositeMeta,
    assign_by: str = "center",
) -> list[str]:
    """
    Map YOLO-format labels (normalized to the original frame) into the
    composite frame. Boxes are routed to the top (global) or bottom (sky)
    panel by centroid, then clipped; tiny boxes after clipping are dropped.
    """
    S = meta.out_size
    ph = meta.panel_h
    W, H = meta.orig_w, meta.orig_h
    out: list[str] = []
    hy = meta.sky_y2

    for line in label_lines:
        parsed = yolo_norm_to_xyxy(line, W, H)
        if parsed is None:
            continue
        c, x1, y1, x2, y2 = parsed
        box = np.array([x1, y1, x2, y2], dtype=np.float64)
        if assign_by == "center":
            cy = 0.5 * (y1 + y2)
            use_bottom = cy < hy
        else:
            raise ValueError(assign_by)

        if use_bottom:
            inter_y2 = min(y2, hy)
            inter_y1 = min(max(y1, 0), inter_y2 - 1e-3)
            if inter_y2 <= inter_y1:
                continue
            bx = clip_xyxy(np.array([x1, inter_y1, x2, inter_y2], dtype=np.float64), W, int(hy))
            bw_ = bx[2] - bx[0]
            bh_ = bx[3] - bx[1]
            if bw_ < 2 or bh_ < 2:
                continue
            lb = xyxy_orig_to_letterbox(bx, meta.bot_r, meta.bot_pad)
            lb[1] += ph
            lb[3] += ph
        else:
            bx = clip_xyxy(box, W, H)
            lb = xyxy_orig_to_letterbox(bx, meta.top_r, meta.top_pad)

        lb = clip_xyxy(lb, S, S)
        if lb[2] - lb[0] < 2 or lb[3] - lb[1] < 2:
            continue
        out.append(xyxy_to_yolo_norm(int(c), lb[0], lb[1], lb[2], lb[3], S, S))
    return out


def composite_xyxy_to_original(xyxy: np.ndarray, meta: CompositeMeta) -> np.ndarray:
    """
    Map a single box (pixel xyxy in composite coordinates) back to original
    image pixel coordinates.
    """
    S = meta.out_size
    ph = meta.panel_h
    x1, y1, x2, y2 = xyxy.astype(np.float64)
    cy = 0.5 * (y1 + y2)
    if cy < ph:
        orig = xyxy_letterbox_to_orig(np.array([x1, y1, x2, y2]), meta.top_r, meta.top_pad)
    else:
        shifted = np.array([x1, y1 - ph, x2, y2 - ph])
        crop = xyxy_letterbox_to_orig(shifted, meta.bot_r, meta.bot_pad)
        orig = crop
    return clip_xyxy(orig, meta.orig_w, meta.orig_h)


def nms_numpy(xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.55) -> list[int]:
    """Greedy NMS; xyxy in pixel coords."""
    if len(xyxy) == 0:
        return []
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def draw_boxes(bgr: np.ndarray, xyxy_list: list[np.ndarray], confs: list[float]) -> np.ndarray:
    out = bgr.copy()
    for xyxy, cf in zip(xyxy_list, confs):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"{cf:.2f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    return out
