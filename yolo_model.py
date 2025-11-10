import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import os

class BoltDetector:

    CLASS_COLORS = {
        1: (0, 140, 255),  # loose_bolt -> orange
        2: (0, 255, 0),    # fixed_bolt -> green
        3: (0, 0, 255)     # no_bolt -> red
    }

    def __init__(self, model_path="color_bolt.pt", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)

    def detect_and_draw(self, frame):
        results = self.model(frame)
        counts = {"loose": 0, "fixed": 0, "no_bolt": 0}

        CLASS_NAME_MAP = {
            "loose_bolt": "loose",
            "fixed_bolt": "fixed",
            "no_bolt": "no_bolt"
        }

        IGNORE_CLASSES = {"hub"}

        h, w = frame.shape[:2]
        scale = w / 1280.0  # baseline for scaling
        font_scale = round(max(0.4, 0.8 * scale), 2)
        box_thickness = max(2, int(3 * scale))  # 👈 slightly thicker boxes
        text_thickness = 1  # keep text sharp

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf < 0.5:
                    continue

                name = self.model.names[cls_id]
                if name in IGNORE_CLASSES:
                    continue

                original_name = self.model.names[cls_id]
                name = CLASS_NAME_MAP.get(original_name, original_name)

                counts[name] = counts.get(name, 0) + 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = self.CLASS_COLORS.get(cls_id, (255, 255, 255))
                label = f"{name} {conf:.2f}"

                # --- Draw thicker bounding box ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

                # --- Text background for clarity ---
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)

                # --- Text overlay ---
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        return frame, counts

    def save_detected_image(self, frame, output_dir="outputs", prefix="detected"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame)
        return path
