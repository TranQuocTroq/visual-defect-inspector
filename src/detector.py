from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class DefectDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Load model tu file weight da train.
        
        model_path     : duong dan den file best.pt
        conf_threshold : nguong tin cay - chi lay box co score > nguong nay
        """
        self.model = YOLO(model_path)
        self.conf  = conf_threshold

    def predict(self, image: np.ndarray) -> list[dict]:
        """
        Nhan anh numpy (BGR), tra ve list cac phat hien.
        Moi phat hien la 1 dict: {box, score, label}
        """
        results = self.model(image, conf=self.conf, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "box"  : box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "score": float(box.conf[0]),
                    "label": "defect"
                })
        
        return detections

    def annotate(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Ve bbox do len anh voi score.
        Tra ve anh da duoc ve bbox.
        """
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            score = det["score"]
            
            # Ve hinh chu nhat do
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Ve label + score
            label = f"defect {score:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return output