# Visual Defect Inspector

Real-time surface defect detection system using YOLOv8 + PatchCore.
Detects manufacturing defects (scratches, cracks, contamination) from images and video.

## Demo
> Live demo link (coming soon)

## Results
| Metric | Value |
|--------|-------|
| mAP50 | 0.783 |
| Precision | 0.856 |
| Recall | 0.690 |
| Inference speed | ~5ms/image (T4 GPU) |

## Project Structure
```
visual-defect-inspector/
├── models/          # Model weights (.pt, .onnx)
├── src/
│   ├── detector.py  # Core detection pipeline
│   ├── api.py       # FastAPI REST endpoint
│   └── ui.py        # Gradio demo interface
├── notebooks/       # Training notebooks (Kaggle)
├── data/samples/    # Sample images for demo
├── tests/           # Unit tests
├── Dockerfile
└── requirements.txt
```

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Run API
uvicorn src.api:app --reload

# Run UI demo
python src/ui.py
```

## Dataset
- MVTec Anomaly Detection dataset
- 15 product categories, 5000+ images
- Training: 1002 defect images
- Validation: 256 defect images
