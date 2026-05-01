---
title: Industrial Visual Inspector (Hybrid AI)
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Industrial Visual Defect Inspector

Real-time surface defect detection for industrial quality control using a two-stage hybrid AI pipeline: **YOLOv8** for macro defect localization and **PatchCore** for micro anomaly detection.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/trong333tn/ai-quality-control)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Dataset](#dataset)
- [Training](#training)
- [Contact](#contact)

---

## Overview

This project implements an automated visual inspection system that detects surface defects in manufactured products in real time. It is designed to replicate the behavior of an industrial conveyor belt quality control station.

**Supported product lines:** Wood · Zipper · Pill (from the MVTec AD benchmark)

**Live demo:** The demo loads one of three sample videos, each containing a realistic mix of good and defective frames (good items significantly outnumber defective ones, as in real production). Every frame is passed through the two-stage pipeline. The interface displays a live analytics report (items scanned, defects blocked, yield rate) and a defect gallery where all rejected frames are captured for human re-inspection.

---

## Architecture

The system is built on a cascade principle: run the fast, lightweight detector first and only invoke the heavier anomaly model when Stage 1 finds nothing. This minimizes average inference latency while maintaining high recall.

```
Input frame
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 — YOLOv8                       │
│  Detects macro defects:                 │
│  scratches, cracks, foreign bodies      │
│  Output: bounding boxes + confidence    │
└──────────────────┬──────────────────────┘
                   │
         Defect detected?
         Yes ──► REJECT  (return immediately)
         No  ──► continue
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 2 — PatchCore                    │
│  Detects micro anomalies:               │
│  texture shifts, contamination,         │
│  subtle defects YOLO cannot localize    │
│  Output: anomaly score + heatmap        │
└──────────────────┬──────────────────────┘
                   │
         Score > threshold?
         Yes ──► REJECT  (heatmap overlay)
         No  ──► PASS
```

**Design rationale**

| Problem | Solution |
|:--------|:---------|
| PatchCore is 8–10× slower than YOLO | Only invoke it when Stage 1 passes, cutting average latency significantly |
| YOLO misses subtle texture-level anomalies | PatchCore is trained on normal images only and flags any statistical deviation |
| PatchCore cannot produce bounding boxes | YOLO handles localization in Stage 1 |
| Single-model systems have higher missed-defect rates | The cascade catches what each individual model misses |

---

## Performance

### Stage 1 — YOLOv8

One model trained per product line. Labels: bounding box annotations on defective images.

| Product | Train Images | Val Images | Precision | Recall | mAP50 |
|:--------|------------:|----------:|----------:|-------:|------:|
| Wood    |          48 |        12 |     0.857 |  0.741 | 0.885 |
| Zipper  |          95 |        24 |     0.913 |  0.872 | 0.914 |
| Pill    |         112 |        29 |     0.866 |  0.897 | 0.922 |

### Stage 2 — PatchCore

Trained unsupervised — only defect-free images are used during training. No defect labels required.

| Product | Image AUROC | Image F1 |
|:--------|------------:|---------:|
| Wood    |      0.9851 |   0.9580 |
| Zipper  |      0.9735 |   0.9791 |
| Pill    |      0.9457 |   0.9489 |

### System throughput

| Metric | Value |
|:-------|:------|
| Stage 1 inference | ~5 ms / frame (NVIDIA T4) |
| Stage 2 inference | ~40 ms / frame (NVIDIA T4) |
| Full pipeline (worst case) | ~45 ms / frame (NVIDIA T4) |
| Effective throughput | ~22 FPS on GPU |

---

## Project Structure

```
visual-defect-inspector/
├── src/
│   ├── detector.py          # Two-stage inference pipeline
│   ├── api.py               # FastAPI REST endpoint
│   └── ui.py                # Gradio web interface
├── models/                  # Model weights (not tracked in git)
│   ├── yolo_wood.pt
│   ├── yolo_zipper.pt
│   ├── yolo_pill.pt
│   ├── patchcore_wood.pt
│   ├── patchcore_zipper.pt
│   └── patchcore_pill.pt
├── notebooks/
│   ├── train_yolo.ipynb
│   └── train_patchcore.ipynb
├── data/
│   └── samples/             # Demo videos (wood.mp4, zipper.mp4, pill.mp4)
├── tests/
├── app.py                   # Entry point: FastAPI + Gradio unified server
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.10+, CUDA GPU recommended.

```bash
git clone https://github.com/TranQuocTroq/visual-defect-inspector.git
cd visual-defect-inspector
pip install -r requirements.txt
```

Place model weight files (`.pt`) in the `models/` directory before running.

---

## Usage

### Run the unified app (API + UI)

```bash
python app.py
```

| Service | URL |
|:--------|:----|
| Web interface | http://localhost:7860 |
| API documentation | http://localhost:7860/docs |

### Run with Docker

```bash
docker build -t defect-inspector .
docker run -p 7860:7860 defect-inspector
```

### Run API server only

```bash
uvicorn src.api:app --reload --port 8000
```

---

## API Reference

### POST /inspect

Submit a product image for automated defect inspection.

**Request** — `multipart/form-data`

| Field | Type | Description |
|:------|:-----|:------------|
| `product` | string | Product line: `wood`, `zipper`, or `pill` |
| `file` | file | Product image (JPEG or PNG) |

**Response** — `application/json`

```json
{
  "product":    "wood",
  "status":     "REJECTED (Stage 2 - PatchCore: 0.61)",
  "is_defect":  true,
  "image_data": "<base64-encoded annotated image>"
}
```

**Status values**

| Value | Meaning |
|:------|:--------|
| `PASS` | No defect detected by either stage |
| `REJECTED (Stage 1 - YOLO: 0.87)` | Macro defect found by YOLOv8 |
| `REJECTED (Stage 2 - PatchCore: 0.61)` | Micro anomaly found by PatchCore |

**Example**

```bash
curl -X POST "http://localhost:7860/inspect" \
  -F "product=wood" \
  -F "file=@sample.jpg"
```

---

## Dataset

[MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) — the standard benchmark for industrial surface defect detection.

- 15 industrial product and texture categories
- 5,000+ high-resolution images with pixel-level ground truth masks
- Categories used: `wood`, `zipper`, `pill`

| Split | Purpose | Size |
|:------|:--------|-----:|
| Normal images — train | PatchCore memory bank | ~200 images / category |
| Normal images — val | Threshold calibration | ~70 images / category |
| Defective images | YOLO training and evaluation | 255 train / 65 val (total across 3 products) |

---

## Training

All training was performed on [Kaggle](https://www.kaggle.com) using a free NVIDIA T4 GPU. Notebooks are provided in the `notebooks/` directory.

### Stage 1 — YOLOv8

- Defect images annotated with bounding boxes
- Data augmentation: horizontal and vertical flips, mosaic, random crop, HSV jitter
- Fine-tuned from `yolov8n.pt` (ImageNet pretrained weights)
- Per-product confidence thresholds calibrated on the validation set to minimize false negatives
- Thresholds: Wood `0.46` · Zipper `0.58` · Pill `0.65`

### Stage 2 — PatchCore (anomalib)

- Fully unsupervised — no defect labels required
- Backbone: WideResNet-50 (layers 2 and 3) for patch feature extraction
- Memory bank constructed from patch-level features of all normal training images
- Coreset subsampling applied to reduce memory bank size while preserving coverage
- Models exported from anomalib as `.pt` files and loaded directly via `torch.load`
- Inference thresholds extracted from `post_processor._image_threshold` in the exported checkpoint
- Thresholds: Wood `0.47` · Zipper `0.50` · Pill `0.60`

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Object detection | YOLOv8 (Ultralytics 8.3.x) |
| Anomaly detection | PatchCore (anomalib 2.3.x) |
| Deep learning runtime | PyTorch + torchvision |
| Image processing | OpenCV 4.10 |
| REST API | FastAPI 0.109 + Uvicorn |
| Web interface | Gradio 4.44 |
| Deployment | Docker on Hugging Face Spaces |
| Training platform | Kaggle (NVIDIA T4 GPU) |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Tran Quoc Trong — [tranquoct157@gmail.com](mailto:tranquoct157@gmail.com)

Project repository: [https://github.com/TranQuocTroq/visual-defect-inspector](https://github.com/TranQuocTroq/visual-defect-inspector)

Live demo: [https://huggingface.co/spaces/trong333tn/ai-quality-control](https://huggingface.co/spaces/trong333tn/ai-quality-control)
