"""
# ======================================
# DETECTOR
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module wraps the trained YOLO model (`best.pt`) behind a clean inference API.
- It converts raw Ultralytics outputs into normalized enterprise detection objects used by tracking, association, rule engine, analytics, and UI modules.
- The rest of the system should not know about YOLO tensor internals; this isolation makes model swaps and ONNX/TensorRT export easier later.

Enterprise architecture reason:
- Production CV products often change models but keep business logic stable.
- A model adapter layer prevents changes in one library from breaking the whole system.
- Each camera worker can own its own detector instance so ByteTrack state does not leak across camera streams.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .config import CONFIG, normalize_class_name
from .logger import get_logger

logger = get_logger("ppe.detector")


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: str
    canonical_class: str
    track_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_dict(self) -> Dict:
        return {
            "bbox": [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)],
            "conf": round(float(self.conf), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "canonical_class": self.canonical_class,
            "track_id": self.track_id,
            "metadata": self.metadata,
        }


class PPEDetector:
    def __init__(self, model_path=CONFIG.MODEL_PATH, tracker_path=CONFIG.TRACKER_CONFIG_PATH):
        self.model_path = str(model_path)
        self.tracker_path = str(tracker_path)
        self._model = None
        self._model_lock = threading.RLock()
        self.names: Dict[int, str] = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self.names = self._extract_names()
            logger.info("Loaded YOLO model from %s with classes: %s", self.model_path, self.names)
        except Exception as exc:
            logger.exception("Failed to load YOLO model: %s", exc)
            raise RuntimeError(
                "Could not load best.pt. Install ultralytics/torch and confirm best.pt is present."
            ) from exc

    def _extract_names(self) -> Dict[int, str]:
        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return {int(k): str(v) for k, v in names.items()}

    def predict_image(self, frame: np.ndarray) -> List[Detection]:
        """Run plain detection for still images where tracking is not meaningful."""
        with self._model_lock:
            results = self._model.predict(
                source=frame,
                conf=CONFIG.CONF_THRESHOLD,
                iou=CONFIG.IOU_THRESHOLD,
                imgsz=CONFIG.IMG_SIZE,
                max_det=CONFIG.MAX_DETECTIONS,
                device=None if CONFIG.DEVICE == "auto" else CONFIG.DEVICE,
                verbose=False,
            )
        return self._parse_results(results)

    def track_frame(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        """Run YOLO + ByteTrack for a frame in a stream.

        `persist=True` keeps tracker state for the detector instance. This is why each live camera
        worker should use a separate PPEDetector instance.
        """
        with self._model_lock:
            results = self._model.track(
                source=frame,
                persist=persist,
                tracker=self.tracker_path,
                conf=CONFIG.CONF_THRESHOLD,
                iou=CONFIG.IOU_THRESHOLD,
                imgsz=CONFIG.IMG_SIZE,
                max_det=CONFIG.MAX_DETECTIONS,
                device=None if CONFIG.DEVICE == "auto" else CONFIG.DEVICE,
                verbose=False,
            )
        return self._parse_results(results)

    def _parse_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
        confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
        clss = boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls).astype(int)

        ids = None
        if getattr(boxes, "id", None) is not None:
            ids = boxes.id.detach().cpu().numpy().astype(int) if hasattr(boxes.id, "detach") else np.asarray(boxes.id).astype(int)

        for i, bbox in enumerate(xyxy):
            cls_id = int(clss[i])
            class_name = self.names.get(cls_id, str(cls_id))
            canonical = normalize_class_name(class_name)
            detections.append(
                Detection(
                    x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]),
                    conf=float(confs[i]), class_id=cls_id, class_name=class_name,
                    canonical_class=canonical, track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return detections


def draw_detections(frame: np.ndarray, detections: Iterable[Detection], violations: Optional[List[Dict]] = None) -> np.ndarray:
    """Annotate frame for output evidence and live stream."""
    out = frame.copy()
    violation_track_ids = {str(v.get("track_id")) for v in (violations or [])}

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        is_violation_person = det.canonical_class == CONFIG.PERSON_CLASS and str(det.track_id) in violation_track_ids
        color = (0, 0, 255) if is_violation_person or det.canonical_class.startswith("no_") else (0, 180, 0)
        if det.canonical_class == CONFIG.PERSON_CLASS:
            color = (0, 0, 255) if is_violation_person else (255, 160, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.canonical_class} {det.conf:.2f}"
        if det.track_id is not None:
            label = f"ID:{det.track_id} {label}"
        cv2.putText(out, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if violations:
        y = 30
        for v in violations[:5]:
            msg = f"VIOLATION: ID {v.get('track_id')} {v.get('violation_type')}"
            cv2.putText(out, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y += 28
    return out
