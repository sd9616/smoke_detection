# UAV smoke detection (YOLO11 + composite tiling)

Quick dry run on **0.1%** of FASDD UAV data. Assumes you unpacked **`FASDD_UAV.zip`** inside this repo:

```
data/FASDD_UAV/
  images/{train,val,test}/
  annotations/YOLO/{train,val,test}/
```

---

## 0. One-time setup

```bash
cd smoke_detection
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Activate the venv in every new terminal:

```bash
source .venv/bin/activate
```

---

## 1. Build 0.1% YOLO subset

Random **0.1%** of labeled images per split (`--sample-frac 0.001`), fixed seed for reproducibility.

```bash
python scripts/prepare_uav_subset.py \
  --fasdd-root data/FASDD_UAV \
  --out data/fasdd_uav_0p1pct \
  --all-images \
  --sample-frac 0.001 \
  --sample-seed 42
```

Output: `data/fasdd_uav_0p1pct/images/`, `labels/`, and `dataset.yaml`.

---

## 2. Build composite training tiles

Jung-style composite (global + sky band) so training matches inference.

```bash
python scripts/build_composite_dataset.py \
  --in-root data/fasdd_uav_0p1pct \
  --out-root data/fasdd_uav_0p1pct_composite \
  --size 640
```

Output: `data/fasdd_uav_0p1pct_composite/dataset.yaml`.

---

## 3. Train (short run for smoke test)

Use a small model and few epochs on this tiny subset; metrics are not meaningful for deployment.

```bash
python scripts/train_yolo.py \
  --data data/fasdd_uav_0p1pct_composite/dataset.yaml \
  --model yolo11n.pt \
  --epochs 5 \
  --imgsz 640 \
  --batch 8 \
  --name smoke_uav_0p1pct
```

Weights: `runs/detect/smoke_uav_0p1pct/weights/best.pt`

Add `--device 0` if you have a CUDA GPU.

---

## 4. Run inference on held-out images

Use original (non-composite) frames, e.g. the test split:

```bash
python scripts/predict_composite.py \
  --weights runs/detect/smoke_uav_0p1pct/weights/best.pt \
  --source data/fasdd_uav_0p1pct/images/test \
  --imgsz 640 \
  --conf 0.08 \
  --out runs/pred_0p1pct_test
```

Visualizations: `runs/pred_0p1pct_test/*.jpg` with boxes and confidence scores.

Optional per-detection text files:

```bash
python scripts/predict_composite.py \
  --weights runs/detect/smoke_uav_0p1pct/weights/best.pt \
  --source data/fasdd_uav_0p1pct/images/test \
  --imgsz 640 \
  --conf 0.08 \
  --out runs/pred_0p1pct_test \
  --save-txt
```

Lower `--conf` (e.g. `0.05`) if you want fewer missed smoke plumes at the cost of more false positives.

---

## Checklist

| Step | Script | Main output |
|------|--------|-------------|
| 1 | `scripts/prepare_uav_subset.py` | `data/fasdd_uav_0p1pct/` |
| 2 | `scripts/build_composite_dataset.py` | `data/fasdd_uav_0p1pct_composite/` |
| 3 | `scripts/train_yolo.py` | `runs/detect/smoke_uav_0p1pct/weights/best.pt` |
| 4 | `scripts/predict_composite.py` | `runs/pred_0p1pct_test/` |

Before training, open `dataset.yaml` and confirm class ids (`0: fire`, `1: smoke`) match a few label `.txt` files.

---

## Full dataset

For production training, repeat steps 1–4 without `--sample-frac`, using e.g. `data/fasdd_uav_yolo` and `data/fasdd_uav_composite`, a larger model (`yolo11s.pt`), and more epochs (`--epochs 100`).
