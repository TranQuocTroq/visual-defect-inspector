from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.detector import DefectDetector

app = FastAPI(title="Visual Defect Inspector API")

# Initialize detector once at server startup
detector = DefectDetector(
    model_path="models/best.pt",
    conf_threshold=0.25
)


@app.get("/health")
def health_check():
    """Check if API server is running."""
    return {"status": "ok"}


@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    """
    Receive an image file, return:
    - defect_count : number of defects detected
    - detections   : list of detected defects with box and score
    - image_base64 : annotated image with bounding boxes (base64 encoded)
    """
    # Read image from request
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    image    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run detection
    detections = detector.predict(image)

    # Draw bounding boxes on image
    annotated = detector.annotate(image, detections)

    # Encode image to base64 for JSON response
    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "defect_count": len(detections),
        "detections"  : detections,
        "image_base64": img_base64
    })