import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from src.detector import DefectDetector

app = FastAPI(title="Industrial Defect Detection API")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
detector = DefectDetector(MODELS_DIR)

# FIX: Hide this endpoint from OpenAPI schema to prevent Gradio parsing crash
@app.post("/inspect", include_in_schema=False)
async def inspect_api(product: str = Form(...), file: UploadFile = File(...)):
    """
    Rest API endpoint for external integration.
    """
    data = await file.read()
    img_np = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    status, result_img, is_defect = detector.inspect(img_bgr, product)
    
    _, buffer = cv2.imencode('.jpg', result_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "product": product,
        "status": status,
        "is_defect": is_defect,
        "image_data": encoded_img
    }