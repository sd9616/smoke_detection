"""Train and run the RGB-thermal YOLO."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.nms import non_max_suppression

from mm_yolo.data import RGBThermalDataset, collate_pairs, list_pairs, load_dataset_yaml, read_bgr, to_chw
from mm_yolo.model import RGBThermalYOLO, resolve_rgb_weights
from smoke_pipeline.composite import build_composite, composite_xyxy_to_original, nms_numpy
from smoke_pipeline.horizon import estimate_horizon_row
from smoke_pipeline.viz import draw_detection


def train(
    data: Path,
    rgb_weights: Path | None,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "",
    workers: int = 4,
    name: str = "mm_yolo_rgbt",
    project: Path = Path("runs/detect"),
    lr: float = 1e-4,
    patience: int = 15,
    freeze_rgb: bool = True,
    freeze_head: bool = True,
    composite: bool = True,
) -> Path:
    spec = load_dataset_yaml(data)
    root = spec["path"]
    train_pairs = list_pairs(root, spec["train"])
    val_pairs = list_pairs(root, spec["val"])
    labeled = sum(1 for *_, label in train_pairs if label.is_file() and label.stat().st_size > 0)
    if labeled == 0:
        raise SystemExit(
            f"No labels in {root / 'labels' / Path(spec['train']).name}. "
            "Add YOLO txt files that share the RGB filename stem."
        )
    names = {int(k): str(v) for k, v in spec.get("names", {0: "smoke"}).items()}
    print(f"RGB-thermal pairs: train {len(train_pairs)} ({labeled} labeled), val {len(val_pairs)}")

    weights = resolve_rgb_weights(rgb_weights)
    model = RGBThermalYOLO.from_checkpoint(weights, freeze_rgb=freeze_rgb, freeze_head=freeze_head)
    if model.nc != len(names):
        raise SystemExit(
            f"Dataset has {len(names)} classes but {weights.name} has nc={model.nc}. "
            "Use the smoke checkpoint with a one-class smoke dataset."
        )
    model.names = names

    dev = _device(device)
    model.to(dev)
    criterion = v8DetectionLoss(model)
    trainable = model.trainable_parameters()
    if not trainable:
        raise SystemExit("Nothing to train. Unfreeze the thermal stream or the head.")
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    frozen = "RGB backbone" + (" and neck/head" if freeze_head else "")
    print(f"Loaded {weights.name}. Frozen: {frozen}. Training {n_train / 1e6:.2f}M / {n_total / 1e6:.2f}M params.")

    train_loader = _loader(root, spec["train"], imgsz, batch, workers, composite, augment=True, shuffle=True)
    val_loader = _loader(root, spec["val"], imgsz, batch, workers, composite, augment=False, shuffle=False)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)

    out_dir = project / name
    weight_dir = out_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "results.csv"
    best_loss = float("inf")
    best_path = weight_dir / "best.pt"
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        totals = torch.zeros(3)
        seen = 0
        for batch_i in train_loader:
            batch_i = _to_device(batch_i, dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                preds = model(batch_i["rgb"], batch_i["thermal"])
                loss, items = criterion(preds, batch_i)
            scaler.scale(loss.sum()).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            scaler.step(optimizer)
            scaler.update()
            totals += items.detach().cpu()
            seen += 1
        val_loss, precision, recall = evaluate(model, criterion, val_loader, dev, use_amp)
        train_loss = (totals / max(seen, 1)).tolist()
        print(
            f"epoch {epoch}/{epochs}  "
            f"train box {train_loss[0]:.4f} cls {train_loss[1]:.4f} dfl {train_loss[2]:.4f}  "
            f"val {val_loss:.4f}  P {precision:.3f}  R {recall:.3f}"
        )
        _append_log(log_path, epoch, train_loss, val_loss, precision, recall)
        _save(weight_dir / "last.pt", model, weights, composite, imgsz, epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            _save(best_path, model, weights, composite, imgsz, epoch)
        else:
            stale += 1
            if stale >= patience:
                print(f"Stopped early at epoch {epoch}. Best val loss {best_loss:.4f}")
                break
    print(f"Best weights: {best_path.resolve()}")
    return best_path


@torch.no_grad()
def evaluate(model, criterion, loader, device, use_amp: bool) -> tuple[float, float, float]:
    """Val loss uses the Detect training output. BatchNorm stays in eval."""
    model.eval()
    total = 0.0
    seen = 0
    hit = 0
    pred_n = 0
    gt_n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        model._force_detect_train = True
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_preds = model(batch["rgb"], batch["thermal"])
            loss, _ = criterion(loss_preds, batch)
        total += float(loss.sum().detach().cpu())
        seen += 1
        model._force_detect_train = False
        infer = model(batch["rgb"], batch["thermal"])
        detections = non_max_suppression(infer, conf_thres=0.15, iou_thres=0.55, max_det=10)
        hits, preds, gts = _score_batch(detections, batch, int(batch["rgb"].shape[-1]))
        hit += hits
        pred_n += preds
        gt_n += gts
    model._force_detect_train = None
    precision = hit / pred_n if pred_n else 0.0
    recall = hit / gt_n if gt_n else 0.0
    return total / max(seen, 1), precision, recall


def predict_dir(
    weights: Path,
    rgb_dir: Path,
    thermal_dir: Path,
    out: Path,
    conf: float = 0.15,
    iou: float = 0.55,
    max_det: int = 5,
    device: str = "",
) -> None:
    dev = _device(device)
    model, meta_cfg = load_trained(weights, dev)
    out.mkdir(parents=True, exist_ok=True)
    rgb_paths = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})
    if not rgb_paths:
        raise SystemExit(f"No images in {rgb_dir}")
    saved = 0
    for rgb_path in rgb_paths:
        thermal_path = _find_thermal(thermal_dir, rgb_path.stem)
        if thermal_path is None:
            print(f"skip {rgb_path.name}: no thermal pair")
            continue
        rgb = read_bgr(rgb_path)
        thermal = read_bgr(thermal_path)
        if thermal.shape[:2] != rgb.shape[:2]:
            thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        boxes = predict_pair(model, rgb, thermal, meta_cfg, dev, conf, iou, max_det)
        canvas = rgb
        for box, score, cls in boxes:
            draw_detection(canvas, box, int(cls), float(score), model.names)
        cv2.imwrite(str(out / rgb_path.name), canvas)
        saved += 1
    print(f"Wrote {saved} images to {out.resolve()}")


def predict_pair(model, rgb, thermal, cfg: dict, device, conf: float, iou: float, max_det: int):
    imgsz = int(cfg.get("imgsz", 640))
    if cfg.get("composite", True):
        horizon = estimate_horizon_row(rgb)
        rgb_in, meta = build_composite(rgb, out_size=imgsz, horizon_y=horizon)
        thermal_in, _ = build_composite(thermal, out_size=imgsz, horizon_y=horizon)
    else:
        meta = None
        rgb_in = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        thermal_in = cv2.resize(thermal, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    batch_rgb = to_chw(rgb_in).unsqueeze(0).to(device)
    batch_thermal = to_chw(thermal_in).unsqueeze(0).to(device)
    model.eval()
    infer = model(batch_rgb, batch_thermal)
    detections = non_max_suppression(infer, conf_thres=conf, iou_thres=iou, max_det=max_det)[0]
    if detections is None or len(detections) == 0:
        return []
    det = detections.detach().cpu().numpy()
    xyxy = det[:, :4]
    if meta is not None:
        mapped = np.stack([composite_xyxy_to_original(box, meta) for box in xyxy], axis=0)
        keep = nms_numpy(mapped, det[:, 4], iou_thresh=iou)
        if max_det > 0:
            keep = keep[:max_det]
        mapped = mapped[keep]
        det = det[keep]
        xyxy = mapped
    return [(xyxy[i], float(det[i, 4]), int(det[i, 5])) for i in range(len(det))]


def load_trained(weights: Path, device: torch.device) -> tuple[RGBThermalYOLO, dict]:
    try:
        blob = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(weights, map_location="cpu")
    model = RGBThermalYOLO.from_scale(
        scale=blob["scale"],
        nc=int(blob["nc"]),
        freeze_rgb=bool(blob.get("freeze_rgb", True)),
        freeze_head=bool(blob.get("freeze_head", True)),
        args=SimpleNamespace(**blob.get("loss_args", {"box": 7.5, "cls": 0.5, "dfl": 1.5})),
    )
    model.load_state_dict(blob["state_dict"])
    model.names = blob.get("names", model.names)
    model.to(device)
    model.eval()
    return model, blob


def _save(path: Path, model: RGBThermalYOLO, rgb_weights: Path, composite: bool, imgsz: int, epoch: int) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scale": model.scale,
            "nc": model.nc,
            "names": model.names,
            "freeze_rgb": model.freeze_rgb,
            "freeze_head": model.freeze_head,
            "composite": composite,
            "imgsz": imgsz,
            "epoch": epoch,
            "rgb_weights": str(rgb_weights),
            "loss_args": {"box": model.args.box, "cls": model.args.cls, "dfl": model.args.dfl},
        },
        path,
    )


def _loader(root, split, imgsz, batch, workers, composite, augment, shuffle) -> DataLoader:
    dataset = RGBThermalDataset(root, split, imgsz=imgsz, composite=composite, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _device(name: str) -> torch.device:
    if name:
        return torch.device(name if name != "0" else "cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _append_log(path: Path, epoch: int, train_loss: list[float], val_loss: float, precision: float, recall: float) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["epoch", "train_box", "train_cls", "train_dfl", "val_loss", "precision", "recall"])
        writer.writerow([epoch, *[f"{v:.6f}" for v in train_loss], f"{val_loss:.6f}", f"{precision:.6f}", f"{recall:.6f}"])


def _score_batch(detections, batch, imgsz: int) -> tuple[int, int, int]:
    gt_cls = batch["cls"].detach().cpu()
    gt_boxes = batch["bboxes"].detach().cpu()
    gt_idx = batch["batch_idx"].detach().cpu().long()
    hits = 0
    preds = 0
    gts = int(gt_boxes.shape[0])
    for image_i, det in enumerate(detections):
        gt = gt_boxes[gt_idx == image_i]
        gts_xyxy = _xywh_to_xyxy(gt, imgsz) if len(gt) else gt
        if det is None or len(det) == 0:
            continue
        pred = det.detach().cpu()[:, :4]
        preds += int(pred.shape[0])
        if len(gts_xyxy) == 0:
            continue
        iou = _iou(pred, gts_xyxy)
        matched: set[int] = set()
        for row in iou:
            best = int(row.argmax())
            if float(row[best]) >= 0.5 and best not in matched:
                matched.add(best)
                hits += 1
    return hits, preds, gts


def _xywh_to_xyxy(boxes: torch.Tensor, imgsz: int) -> torch.Tensor:
    xyxy = boxes.clone()
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * imgsz
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * imgsz
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * imgsz
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * imgsz
    return xyxy


def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _find_thermal(directory: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None
