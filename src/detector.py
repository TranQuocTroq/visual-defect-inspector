from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class DefectDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Load model from pretrained weight file.

        model_path     : path to best.pt file
        conf_threshold : confidence threshold - only keep boxes above this score
        """
        self.model = YOLO(model_path)
        self.conf  = conf_threshold

    def predict(self, image: np.ndarray) -> list[dict]:
        """
        Receive image as numpy array (BGR), return list of detections.
        Each detection is a dict: {box, score, label}
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
        Draw bounding boxes on image with confidence score.
        Return annotated image as numpy array.
        """
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            score = det["score"]

            # Draw red rectangle around defect area
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw label with confidence score
            label = f"defect {score:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return output