import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.detector import DefectDetector

# Khoi tao detector
detector = DefectDetector(
    model_path="models/best.pt",
    conf_threshold=0.25
)
print("Load model thanh cong!")

# Doc anh
img_path = "data/samples/test.png"
image = cv2.imread(img_path)
print(f"Anh size: {image.shape}")

# Chay predict
detections = detector.predict(image)
print(f"\nPhat hien duoc {len(detections)} loi:")
for i, det in enumerate(detections):
    box = [round(v) for v in det['box']]
    print(f"  Loi {i+1}: box={box}, score={det['score']:.2f}")

# Ve bbox
output = detector.annotate(image, detections)

# Luu ket qua
out_path = "data/samples/result.png"
cv2.imwrite(out_path, output)
print(f"\nDa luu ket qua: {out_path}")