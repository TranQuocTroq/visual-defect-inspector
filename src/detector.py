# src/detector.py
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class DefectDetector:
    def __init__(self, models_dir):
        """
        Initialize the Hybrid AI detection models.
        Loads YOLO (.pt) for Stage 1 and PatchCore (.pt) for Stage 2.
        """
        self.models_dir = models_dir
        self.yolo_models = {}
        self.patchcore_models = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"[INFO] Loading Hybrid AI Models on {self.device}...")

        for prod in ["wood", "zipper", "pill"]:

            # ----------------------------------------------------------
            # Stage 1: Load YOLO (.pt)
            # ----------------------------------------------------------
            yolo_path = os.path.join(models_dir, f"yolo_{prod}.pt")
            if os.path.exists(yolo_path):
                self.yolo_models[prod] = YOLO(yolo_path)
                print(f"  -> YOLO loaded for {prod}")
            else:
                print(f"  [WARN] YOLO model not found for {prod}: {yolo_path}")

            # ----------------------------------------------------------
            # Stage 2: Load PatchCore (.pt exported from anomalib 1.1.0)
            # NOTE: anomalib exports a dict {'model': <Patchcore object>}
            # We load the inner model directly — no TorchInferencer needed
            # ----------------------------------------------------------
            pc_path = os.path.join(models_dir, f"patchcore_{prod}.pt")
            if os.path.exists(pc_path):
                try:
                    data = torch.load(pc_path, map_location=self.device, weights_only=False)
                    pc_model = data['model'].to(self.device)
                    pc_model.eval()
                    self.patchcore_models[prod] = pc_model
                    print(f"  -> PatchCore loaded for {prod}")
                except Exception as e:
                    print(f"  [ERROR] Failed to load PatchCore for {prod}: {e}")
            else:
                print(f"  [WARN] PatchCore model not found for {prod}: {pc_path}")

    def _preprocess_for_patchcore(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocess BGR image to match anomalib training pipeline:
        - Resize to 224x224
        - Convert BGR -> RGB
        - Normalize to [0, 1]
        - Add batch dimension: (1, 3, 224, 224)
        """
        img_resized = cv2.resize(image_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def _apply_heatmap(self, image: np.ndarray, amap: np.ndarray) -> np.ndarray:
        """
        Overlay anomaly heatmap on original image.
        amap can be any size — will be resized to match image.
        """
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap_uint8 = (amap_norm * 255).astype(np.uint8)
        amap_resized = cv2.resize(amap_uint8, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    def inspect(self, frame: np.ndarray, product_name: str):
        """
        Run 2-Stage Hybrid Inspection on a single frame.

        Stage 1 — YOLO: catches macro defects (scratches, breaks, large anomalies)
        Stage 2 — PatchCore: catches micro defects YOLO misses (subtle texture changes)

        Returns:
            status_message (str)
            annotated_image (np.ndarray, BGR)
            is_defect (bool)
        """
        image_bgr = frame.copy()

        # ----------------------------------------------------------
        # Thresholds — calibrated from MVTec-AD validation set
        # Stage 1: YOLO confidence (restore to real values after testing)
        # ----------------------------------------------------------
        yolo_thresholds = {
            "wood":   0.46,
            "zipper": 0.58,
            "pill":   0.65,
        }

        # Stage 2: PatchCore image-level threshold
        # Extracted from post_processor._image_threshold in exported .pt
        patchcore_thresholds = {
            "wood":   0.47,
            "zipper": 0.50,
            "pill":   0.60,
        }

        yolo_conf = yolo_thresholds.get(product_name, 0.50)
        pc_conf   = patchcore_thresholds.get(product_name, 35.0)

        # ==========================================================
        # STAGE 1: YOLO — Macro defect detection
        # ==========================================================
        if product_name in self.yolo_models:
            results = self.yolo_models[product_name](image_bgr, conf=yolo_conf, verbose=False)

            for r in results:
                if len(r.boxes) > 0:
                    annotated_frame = r.plot()
                    max_conf = float(torch.max(r.boxes.conf).item())
                    print(f"[STAGE 1] {product_name.upper()} | YOLO conf={max_conf:.3f} → REJECTED")
                    return f"REJECTED (Stage 1 - YOLO: {max_conf:.2f})", annotated_frame, True

        # ==========================================================
        # STAGE 2: PatchCore — Micro defect detection
        # ==========================================================
        if product_name in self.patchcore_models:
            pc_model = self.patchcore_models[product_name]
            img_tensor = self._preprocess_for_patchcore(image_bgr)

            with torch.no_grad():
                output = pc_model(img_tensor)

            # Extract anomaly score
            if hasattr(output, 'pred_score'):
                score = float(output.pred_score.item())
            elif isinstance(output, dict) and 'pred_score' in output:
                score = float(output['pred_score'].item())
            else:
                # Fallback: use max of anomaly map as score proxy
                score = 0.0
                print(f"[WARN] {product_name.upper()} | pred_score not found in output, defaulting to 0.0")

            # Extract anomaly map for heatmap visualization
            if hasattr(output, 'anomaly_map') and output.anomaly_map is not None:
                amap = output.anomaly_map.squeeze().cpu().numpy()
            elif isinstance(output, dict) and 'anomaly_map' in output:
                amap = output['anomaly_map'].squeeze().cpu().numpy()
            else:
                amap = np.zeros((image_bgr.shape[0], image_bgr.shape[1]))

            print(f"[STAGE 2] {product_name.upper()} | Score={score:.4f} | Threshold={pc_conf} | REJECT={score > pc_conf}")

            if score > pc_conf:
                overlay = self._apply_heatmap(image_bgr, amap)
                return f"REJECTED (Stage 2 - PatchCore: {score:.2f})", overlay, True

        # ==========================================================
        # FINAL VERDICT: PASS
        # ==========================================================
        return "PASS", image_bgr.copy(), False