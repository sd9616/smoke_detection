#!/usr/bin/env python3
"""
Preview the kind of outputs this pipeline produces before a long training run.

Writes a small gallery under --out (default runs/safety_check/) with:
  - Ground-truth smoke boxes on original frames (what labels look like)
  - Optional model predictions (what inference looks like after training)
  - Optional composite tiles + labels (what the detector is trained on)

Example (labels only, after prepare):
  python scripts/safety_check_preview.py --data-root data/fasdd_smoke_half

Example (compare to an existing checkpoint):
  python scripts/safety_check_preview.py \\
    --data-root data/fasdd_smoke_half \\
    --weights runs/detect/smoke_half_50ep/weights/best.pt \\
    --conf 0.12 --max-det 3

Example (include composite training view):
  python scripts/safety_check_preview.py \\
    --data-root data/fasdd_smoke_half \\
    --composite-root data/fasdd_smoke_half_composite
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from smoke_pipeline.composite import build_composite, composite_xyxy_to_original, nms_numpy, yolo_norm_to_xyxy
from smoke_pipeline.fasdd_paths import list_images
from smoke_pipeline.viz import draw_detection, resolve_class_name


def load_class_names(data_root: Path) -> dict[int, str]:
    yaml_path = data_root / "dataset.yaml"
    names: dict[int, str] = {0: "smoke"}
    if not yaml_path.is_file():
        return names
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == "names:":
            in_names = True
            names.clear()
            continue
        if in_names and ":" in s:
            idx, label = s.split(":", 1)
            names[int(idx.strip())] = label.strip()
    return names


def read_gt_boxes(lab_path: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    if not lab_path.is_file():
        return []
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        parsed = yolo_norm_to_xyxy(line, w, h)
        if parsed is None:
            continue
        c, x1, y1, x2, y2 = parsed
        boxes.append((c, (int(x1), int(y1), int(x2), int(y2))))
    return boxes


def draw_gt_panel(bgr: np.ndarray, boxes: list[tuple[int, tuple[int, int, int, int]]], names: dict[int, str]) -> np.ndarray:
    out = bgr.copy()
    for cls_id, xyxy in boxes:
        cname = resolve_class_name(names, cls_id)
        draw_detection(out, xyxy, cls_id, 1.0, {cls_id: f"{cname} GT"})
    return out


def draw_pred_panel(
    bgr: np.ndarray,
    detections: list[tuple[int, float, tuple[int, int, int, int]]],
    class_names: dict[int, str] | list[str],
) -> np.ndarray:
    out = bgr.copy()
    for cls_id, conf, xyxy in detections:
        draw_detection(out, xyxy, cls_id, conf, class_names)
    if not detections:
        cv2.putText(
            out,
            "no detections",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def stack_horizontal(panels: list[np.ndarray], gap: int = 8) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.zeros((h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)
        if gap:
            padded.append(np.full((h, gap, 3), 32, dtype=np.uint8))
    return np.hstack(padded[:-1])


def add_title(bar_h: int, w: int, text: str) -> np.ndarray:
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def titled_panel(bgr: np.ndarray, title: str) -> np.ndarray:
    bar = add_title(28, bgr.shape[1], title)
    return np.vstack([bar, bgr])


def collect_splits(data_root: Path, splits: list[str] | None) -> list[str]:
    img_root = data_root / "images"
    if splits:
        return [s for s in splits if (img_root / s).is_dir()]
    return sorted(d.name for d in img_root.iterdir() if d.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview GT / pred / composite outputs for a safety check.")
    ap.add_argument("--data-root", type=Path, required=True, help="Prepared YOLO root (images/, labels/)")
    ap.add_argument("--composite-root", type=Path, default=None, help="Optional composite dataset root")
    ap.add_argument("--weights", type=Path, default=None, help="Optional .pt weights for pred preview")
    ap.add_argument("--out", type=Path, default=Path("runs/safety_check"))
    ap.add_argument("--splits", nargs="*", default=["val", "train"])
    ap.add_argument("--samples-per-split", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--max-det", type=int, default=3)
    args = ap.parse_args()

    data_root = args.data_root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = load_class_names(data_root)
    rng = random.Random(args.seed)

    model = None
    model_names: dict[int, str] | list[str] = names
    if args.weights is not None:
        from ultralytics import YOLO

        model = YOLO(str(args.weights.expanduser().resolve()))
        model_names = model.names

    lines: list[str] = [
        "# Smoke pipeline safety check",
        f"data_root: {data_root}",
        f"composite_root: {args.composite_root or '(none)'}",
        f"weights: {args.weights or '(none — GT only)'}",
        f"classes: {names}",
        "",
    ]

    for split in collect_splits(data_root, args.splits):
        idir = data_root / "images" / split
        ldir = data_root / "labels" / split
        imgs = list_images(idir)
        if not imgs:
            lines.append(f"{split}: no images")
            continue
        k = min(args.samples_per_split, len(imgs))
        picked = rng.sample(imgs, k=k)
        lines.append(f"{split}: {len(imgs)} images, previewing {k}")

        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for im_path in picked:
            bgr = cv2.imread(str(im_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            lab_path = ldir / f"{im_path.stem}.txt"
            gt_boxes = read_gt_boxes(lab_path, w, h)
            panels: list[np.ndarray] = [titled_panel(draw_gt_panel(bgr, gt_boxes, names), "ground truth")]

            if args.composite_root is not None:
                comp_im = args.composite_root.expanduser().resolve() / "images" / split / f"{im_path.stem}.jpg"
                comp_lab = args.composite_root / "labels" / split / f"{im_path.stem}.txt"
                if comp_im.is_file():
                    comp_bgr = cv2.imread(str(comp_im))
                    if comp_bgr is not None:
                        ch, cw = comp_bgr.shape[:2]
                        comp_gt = read_gt_boxes(comp_lab, cw, ch)
                        panels.append(
                            titled_panel(draw_gt_panel(comp_bgr, comp_gt, names), "composite train view")
                        )

            if model is not None:
                comp, meta = build_composite(bgr, out_size=args.imgsz)
                res = model.predict(source=comp, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                dets: list[tuple[int, float, tuple[int, int, int, int]]] = []
                if res.boxes is not None and len(res.boxes):
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    mapped = np.stack([composite_xyxy_to_original(b, meta) for b in xyxy], axis=0)
                    keep = nms_numpy(mapped, scores, iou_thresh=args.iou)
                    if args.max_det > 0 and len(keep) > args.max_det:
                        order = np.argsort(scores[keep])[::-1][: args.max_det]
                        keep = [keep[i] for i in order]
                    for i in keep:
                        dets.append((int(cls[i]), float(scores[i]), tuple(int(v) for v in mapped[i])))
                panels.append(
                    titled_panel(draw_pred_panel(bgr, dets, model_names), f"prediction conf>={args.conf}")
                )

            grid = stack_horizontal(panels)
            out_path = split_dir / f"{im_path.stem}_safety.jpg"
            cv2.imwrite(str(out_path), grid)
            lines.append(f"  - {out_path.name}: {len(gt_boxes)} GT box(es)")

    report = out / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Smoke safety check</title>",
        "<style>body{font-family:sans-serif;margin:1rem;background:#111;color:#eee}",
        "img{max-width:100%;margin:0.5rem 0;border:1px solid #444}",
        "h2{margin-top:2rem}</style></head><body>",
        "<h1>Smoke pipeline safety check</h1>",
        "<p>Open this file after running <code>scripts/safety_check_preview.py</code>.</p>",
        "<ul>",
        "<li><b>Left:</b> ground-truth smoke boxes (what training targets look like)</li>",
        "<li><b>Middle (optional):</b> composite tile + labels (what YOLO trains on)</li>",
        "<li><b>Right (optional):</b> model predictions on the original frame (what you deploy)</li>",
        "</ul>",
    ]
    for split in collect_splits(data_root, args.splits):
        split_dir = out / split
        shots = sorted(split_dir.glob("*_safety.jpg"))
        if not shots:
            continue
        html_lines.append(f"<h2>{split}</h2>")
        for p in shots:
            rel = p.relative_to(out).as_posix()
            html_lines.append(f'<p><code>{p.name}</code></p><img src="{rel}" alt="{p.name}">')
    html_lines.append("</body></html>")
    (out / "index.html").write_text("\n".join(html_lines), encoding="utf-8")

    print(f"wrote previews under {out}")
    print(f"  report: {report}")
    print(f"  gallery: {out / 'index.html'}")
    print("Open index.html in a browser to review before a long training run.")


if __name__ == "__main__":
    main()
