# UAV smoke detection (YOLO11 + composite tiling)

Assumes you unpacked **`FASDD_UAV.zip`** under `data/FASDD_UAV/`:

```
data/FASDD_UAV/
  images/*.jpg
  annotations/YOLO_UAV/labels/*.txt
  annotations/YOLO_UAV/{train,val,test}.txt
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

Add `--device 0` on training/inference when a CUDA GPU is available.

---

## Safety check preview (recommended)

Before a long train, generate a small gallery that shows **what you should expect**:

| Panel | Meaning |
|-------|---------|
| **ground truth** | Human smoke boxes on the original UAV frame |
| **composite train view** | The 640×640 tile YOLO actually trains on (optional) |
| **prediction** | Boxes labeled `smoke 0.xx` on the original frame (optional; needs weights) |

**After prepare** (labels + composites only — no GPU training yet):

```bash
python scripts/safety_check_preview.py \
  --data-root data/fasdd_smoke_half \
  --composite-root data/fasdd_smoke_half_composite \
  --samples-per-split 6
```

**After training** (add weights to preview inference-style output):

```bash
python scripts/safety_check_preview.py \
  --data-root data/fasdd_smoke_half \
  --composite-root data/fasdd_smoke_half_composite \
  --weights runs/detect/smoke_half_50ep/weights/best.pt \
  --conf 0.12 \
  --max-det 3 \
  --samples-per-split 6
```

Open **`runs/safety_check/index.html`** in a browser (also see `runs/safety_check/REPORT.md`).

What to look for:

- GT panels should show **one or a few** cyan **smoke GT** boxes per scene, not hundreds.
- Prediction panels should show **`smoke`** text (not `0:0.01`); use `--max-det 1` in `predict_composite.py` if you only want the top detection.
- If predictions are empty, try lower `--conf` on a small model, or finish more training before trusting the check.

---

## 1. Build 0.1% YOLO subset (quick smoke test)

Random **0.1%** of labeled images per split (`--sample-frac 0.001`), fixed seed for reproducibility.

```bash
python scripts/prepare_uav_subset.py \
  --fasdd-root data/FASDD_UAV \
  --out data/fasdd_smoke_0p1pct \
  --all-images \
  --smoke-only \
  --sample-frac 0.001 \
  --sample-seed 42
```

Output: `data/fasdd_smoke_0p1pct/images/`, `labels/`, and `dataset.yaml` (class `0: smoke` only).

---

## 2. Build composite training tiles

Jung-style composite (global + sky band) so training matches inference.

```bash
python scripts/build_composite_dataset.py \
  --in-root data/fasdd_smoke_0p1pct \
  --out-root data/fasdd_smoke_0p1pct_composite \
  --size 640
```

Output: `data/fasdd_smoke_0p1pct_composite/dataset.yaml`.

---

## 3. Train (short run for smoke test)

Use a small model and few epochs on this tiny subset; metrics are not meaningful for deployment.

```bash
python scripts/train_yolo.py \
  --data data/fasdd_smoke_0p1pct_composite/dataset.yaml \
  --model yolo11n.pt \
  --epochs 25 \
  --imgsz 640 \
  --batch 8 \
  --name smoke_0p1pct_test \
  --val-conf 0.01 \
  --device 0
```

Weights: `runs/detect/smoke_0p1pct_test/weights/best.pt`

---

## 4. Run inference on held-out images

Use **original** (non-composite) frames — do not pass `*_composite/images/` into `predict_composite.py` (it will tile twice).

```bash
python scripts/predict_composite.py \
  --weights runs/detect/smoke_0p1pct_test/weights/best.pt \
  --source data/fasdd_smoke_0p1pct/images/val \
  --imgsz 640 \
  --conf 0.01 \
  --max-det 3 \
  --out runs/pred_smoke_0p1pct_val
```

Boxes are labeled **`smoke`** with confidence (e.g. `smoke 0.12`). Raise `--conf` after training on more data; use `--max-det 1` for a single best box per image.

Optional per-detection text files:

```bash
python scripts/predict_composite.py \
  --weights runs/detect/smoke_0p1pct_test/weights/best.pt \
  --source data/fasdd_smoke_0p1pct/images/test \
  --imgsz 640 \
  --conf 0.01 \
  --max-det 3 \
  --out runs/pred_smoke_0p1pct_test \
  --save-txt
```

---

## 50% smoke-only run (recommended before full training)

Random **50%** of smoke-labeled images per split, **50 epochs**, `yolo11s`. Expect on the order of **~3–6 hours** end-to-end on a mid-range GPU (prepare + composite + train).

### 1. Prepare 50% subset

```bash
python scripts/prepare_uav_subset.py \
  --fasdd-root data/FASDD_UAV \
  --out data/fasdd_smoke_half \
  --all-images \
  --smoke-only \
  --sample-frac 0.5 \
  --sample-seed 42
```

### 2. Build composites

```bash
python scripts/build_composite_dataset.py \
  --in-root data/fasdd_smoke_half \
  --out-root data/fasdd_smoke_half_composite \
  --size 640
```

### 3. Train (50 epochs)

```bash
python scripts/train_yolo.py \
  --data data/fasdd_smoke_half_composite/dataset.yaml \
  --model yolo11s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --name smoke_half_50ep \
  --val-conf 0.15 \
  --device 0
```

Weights: `runs/detect/smoke_half_50ep/weights/best.pt`

Use `--batch 8` if GPU memory is limited (e.g. 8 GB).

### 4. Safety check + inference

Preview val/train samples (same style as final deployment):

```bash
python scripts/safety_check_preview.py \
  --data-root data/fasdd_smoke_half \
  --composite-root data/fasdd_smoke_half_composite \
  --weights runs/detect/smoke_half_50ep/weights/best.pt \
  --conf 0.12 \
  --max-det 3
```

Full val folder export:

```bash
python scripts/predict_composite.py \
  --weights runs/detect/smoke_half_50ep/weights/best.pt \
  --source data/fasdd_smoke_half/images/val \
  --imgsz 640 \
  --conf 0.12 \
  --max-det 3 \
  --out runs/pred_smoke_half_val
```

Tune `--conf` (e.g. `0.08–0.25`) on val to balance missed smoke vs. false positives.

---

## Checklist

| Step | Script | 0.1% output | 50% output |
|------|--------|-------------|------------|
| 1 | `scripts/prepare_uav_subset.py` | `data/fasdd_smoke_0p1pct/` | `data/fasdd_smoke_half/` |
| 2 | `scripts/build_composite_dataset.py` | `data/fasdd_smoke_0p1pct_composite/` | `data/fasdd_smoke_half_composite/` |
| 3 | `scripts/train_yolo.py` | `runs/detect/smoke_0p1pct_test/weights/best.pt` | `runs/detect/smoke_half_50ep/weights/best.pt` |
| Safety | `scripts/safety_check_preview.py` | `runs/safety_check/` | `runs/safety_check/` |
| 4 | `scripts/predict_composite.py` | `runs/pred_smoke_0p1pct_val/` | `runs/pred_smoke_half_val/` |

`--smoke-only` keeps FASDD class `1` (smoke) and drops fire boxes; labels use class `0: smoke` in `dataset.yaml`.

---

## Full dataset

For production training, repeat the 50% steps without `--sample-frac`, using e.g. `data/fasdd_smoke_yolo` and `data/fasdd_smoke_composite`, `--epochs 100`, and the same `--smoke-only` flag.

```bash
python scripts/prepare_uav_subset.py \
  --fasdd-root data/FASDD_UAV \
  --out data/fasdd_smoke_yolo \
  --all-images \
  --smoke-only

python scripts/build_composite_dataset.py \
  --in-root data/fasdd_smoke_yolo \
  --out-root data/fasdd_smoke_composite \
  --size 640

python scripts/train_yolo.py \
  --data data/fasdd_smoke_composite/dataset.yaml \
  --model yolo11s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name smoke_full_100ep \
  --val-conf 0.15 \
  --device 0
```

---

## Pyro-SDIS (Hugging Face, fixed cameras + composite crops)

[pyronear/pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) — ~33.6k frames from Pyronear wildfire cameras (train/val, single-class smoke). Images are ~1280px; smoke boxes are often small and in the upper sky band.

This repo uses the same **composite cropping** as FASDD UAV training: top half = full-frame letterbox, bottom half = sky crop above an estimated horizon — so YOLO trains on enlarged smoke regions without a second forward pass at inference.

### 0. Extra dependencies

```bash
pip install datasets huggingface_hub
```

### 1. Download and export YOLO layout (~3.3 GB)

```bash
python scripts/prepare_pyro_sdis.py --out data/pyro_sdis_yolo
```

Quick test (20 images per split):

```bash
python scripts/prepare_pyro_sdis.py --out data/pyro_sdis_yolo_tiny --max-samples 20
```

Optional: only frames with at least one smoke box (~28k images):

```bash
python scripts/prepare_pyro_sdis.py --out data/pyro_sdis_yolo --labeled-only
```

### 2. Build composite training tiles

```bash
python scripts/build_composite_dataset.py \
  --in-root data/pyro_sdis_yolo \
  --out-root data/pyro_sdis_yolo_composite \
  --size 640
```

### 3. Train YOLO11

```bash
python scripts/train_pyro_sdis.py \
  --data data/pyro_sdis_yolo_composite/dataset.yaml \
  --epochs 50 \
  --batch 16 \
  --name pyro_sdis_composite \
  --device 0
```

Or use `scripts/train_yolo.py` directly with the same `--data` path.

Weights: `runs/detect/pyro_sdis_composite/weights/best.pt`

### 4. Inference on val frames

Use **original** images (not composite tiles):

```bash
python scripts/predict_composite.py \
  --weights runs/detect/pyro_sdis_composite/weights/best.pt \
  --source data/pyro_sdis_yolo/images/val \
  --conf 0.12 \
  --max-det 3 \
  --out runs/pred_pyro_sdis_val
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/prepare_pyro_sdis.py` | `data/pyro_sdis_yolo/` |
| 2 | `scripts/build_composite_dataset.py` | `data/pyro_sdis_yolo_composite/` |
| 3 | `scripts/train_pyro_sdis.py` | `runs/detect/pyro_sdis_composite/weights/best.pt` |
| 4 | `scripts/predict_composite.py` | `runs/pred_pyro_sdis_val/` |

---

## RT-DETR vs YOLO11 on FASDD_UAV (same dataset)

Train RT-DETR on the **same `dataset.yaml`** as YOLO11 so val mAP / precision / recall are comparable.

### 1. Use the same prepared data as your YOLO run

If you already trained `smoke_full_100ep`, reuse that composite export:

```text
data/fasdd_smoke_composite/dataset.yaml   # or fasdd_smoke_half_composite, etc.
```

If not prepared yet, follow [Full dataset](#full-dataset) (prepare → composite).

### 2. Train RT-DETR (matched epochs / resolution)

From the repo root (`smoke_detection/`):

```bash
python scripts/train_rtdetr.py \
  --data data/fasdd_smoke_composite/dataset.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --name rtdetr_l_fasdd_full \
  --val-conf 0.15 \
  --device 0
```

Weights: `runs/detect/rtdetr_l_fasdd_full/weights/best.pt`

Use `--batch 8` (or lower) if GPU memory is tight — RT-DETR is heavier than YOLO11s.

### 3. Compare metrics

| Model | Script | Typical results path |
|-------|--------|----------------------|
| YOLO11 | `scripts/train_yolo.py` | `runs/detect/smoke_full_100ep/` |
| RT-DETR | `scripts/train_rtdetr.py` | `runs/detect/rtdetr_l_fasdd_full/` |

Compare `results.csv` and `results.png` in each run folder (mAP50, mAP50-95, precision, recall).

Optional: build a report like the YOLO full run:

```bash
mkdir -p artifacts/rtdetr_l_fasdd_full
ln -sf ../../runs/detect/rtdetr_l_fasdd_full/results.csv artifacts/rtdetr_l_fasdd_full/
ln -sf ../../runs/detect/rtdetr_l_fasdd_full/weights/best.pt artifacts/rtdetr_l_fasdd_full/
python scripts/eval_artifacts.py --artifacts artifacts/rtdetr_l_fasdd_full \
  --image-dir data/fasdd_smoke_yolo/images/val
```

### Notes

- **Composite tiles:** YOLO11 in this repo trains on composite 640×640 tiles. Point RT-DETR at `*_composite/dataset.yaml` for the same training views.
- **Raw frames (optional):** To compare both models on original UAV frames without composite, use `data/fasdd_smoke_yolo/dataset.yaml` for **both** `train_yolo.py` and `train_rtdetr.py` (no `build_composite_dataset.py`).
- **Inference:** YOLO uses `predict_composite.py` on original images. RT-DETR val metrics during training are on the same composite val split; for deployment on full frames use Ultralytics `RTDETR.predict` on `data/fasdd_smoke_yolo/images/val` or add a small predict script later.

The copy under `fire_detection_code/scripts/train_rtdetr.py` defaults to Pyro-SDIS; pass `--data` to point at FASDD:

```bash
python fire_detection_code/scripts/train_rtdetr.py \
  --data data/fasdd_smoke_composite/dataset.yaml \
  --project runs/detect --name rtdetr_l_fasdd_full
```

---

## Two models

These stay separate. YOLO is the model for the raw frames. RT-DETR is the attention model, trained on the prepared images.

| Model | Data | Train | Weights |
|-------|------|-------|---------|
| YOLO11s | Raw frames. The finished run is reused. | not run again | `runs/detect/smoke_full_100ep/weights/best.pt` |
| Attention (RT-DETR-L) | Composite images (`data/fasdd_smoke_composite`) | `scripts/train_attention.py` | `runs/detect/rtdetr_l_fasdd_full/weights/best.pt`, or `runs/detect/attention_images/weights/best.pt` after a new run |

If the YOLO checkpoint is already on the machine, `train_attention.py` does not train it again. It only trains RT-DETR, and it skips that too when an attention checkpoint is already present.

```bash
python scripts/train_attention.py \
  --data data/fasdd_smoke_composite/dataset.yaml \
  --epochs 100 \
  --batch 8 \
  --device 0
```

Run YOLO on raw frames with `scripts/predict_composite.py`. Run the attention model on the images:

```bash
python scripts/predict_attention.py \
  --source data/fasdd_smoke_composite/images/val \
  --conf 0.15 \
  --out runs/pred_attention
```

Pass `--force` to `train_attention.py` only when you want a new attention run on top of an existing checkpoint.
