"""
# ======================================
# DETECTOR
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module wraps the trained YOLO model (`best.pt`) behind a clean inference API.
- It converts raw Ultralytics outputs into normalized enterprise detection objects used by:
    - tracking
    - association
    - compliance engine
    - analytics
    - UI
- The rest of the system should not know about YOLO tensor internals.
- This isolation makes:
    - model swaps
    - TensorRT export
    - ONNX deployment
    - future upgrades
  much easier.

Enterprise architecture reason:
- Production CV systems frequently swap models.
- The business logic layer should remain stable.
- Each camera owns an isolated detector instance so ByteTrack state does not leak across streams.
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


# ======================================
# DETECTION OBJECT
# ======================================

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

    # ======================================
    # BBOX
    # ======================================

    @property
    def bbox(self) -> Tuple[float, float, float, float]:

        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2
        )

    # ======================================
    # AREA
    # ======================================

    @property
    def area(self) -> float:

        return (

            max(0.0, self.x2 - self.x1)

            *

            max(0.0, self.y2 - self.y1)

        )

    # ======================================
    # CENTER
    # ======================================

    @property
    def center(self) -> Tuple[float, float]:

        return (

            (self.x1 + self.x2) / 2.0,

            (self.y1 + self.y2) / 2.0

        )

    # ======================================
    # SERIALIZATION
    # ======================================

    def as_dict(self) -> Dict:

        return {

            "bbox": [

                round(self.x1, 2),

                round(self.y1, 2),

                round(self.x2, 2),

                round(self.y2, 2)

            ],

            "conf": round(float(self.conf), 4),

            "class_id": int(self.class_id),

            "class_name": self.class_name,

            "canonical_class": self.canonical_class,

            "track_id": self.track_id,

            "metadata": self.metadata,
        }


# ======================================
# PPE DETECTOR
# ======================================

class PPEDetector:

    # ======================================
    # INIT
    # ======================================

    def __init__(

        self,

        model_path=CONFIG.MODEL_PATH,

        tracker_path=CONFIG.TRACKER_CONFIG_PATH

    ):

        self.model_path = str(model_path)

        self.tracker_path = str(tracker_path)

        self._model = None

        self._model_lock = threading.RLock()

        self.names: Dict[int, str] = {}

        self._load_model()

    # ======================================
    # LOAD MODEL
    # ======================================

    def _load_model(self) -> None:

        try:

            from ultralytics import YOLO

            self._model = YOLO(
                self.model_path
            )

            self.names = (
                self._extract_names()
            )

            logger.info(

                "Loaded YOLO model from %s with classes: %s",

                self.model_path,

                self.names

            )

        except Exception as exc:

            logger.exception(

                "Failed to load YOLO model: %s",

                exc

            )

            raise RuntimeError(

                "Could not load best.pt. "
                "Install ultralytics/torch "
                "and confirm best.pt exists."

            ) from exc

    # ======================================
    # EXTRACT CLASS NAMES
    # ======================================

    def _extract_names(self) -> Dict[int, str]:

        names = (
            getattr(self._model, "names", {})
            or
            {}
        )

        if isinstance(names, list):

            return {

                i: n

                for i, n in enumerate(names)

            }

        return {

            int(k): str(v)

            for k, v in names.items()

        }

    # ======================================
    # IMAGE DETECTION
    # ======================================

    def predict_image(
        self,
        frame: np.ndarray
    ) -> List[Detection]:

        """
        Plain detection for:
        - images
        - snapshots
        - offline analysis
        """

        with self._model_lock:

            results = self._model.predict(

                source=frame,

                conf=CONFIG.CONF_THRESHOLD,

                iou=CONFIG.IOU_THRESHOLD,

                imgsz=CONFIG.IMG_SIZE,

                max_det=CONFIG.MAX_DETECTIONS,

                device=(
                    None
                    if CONFIG.DEVICE == "auto"
                    else CONFIG.DEVICE
                ),

                verbose=False
            )

        return self._parse_results(results)

    # ======================================
    # VIDEO TRACKING
    # ======================================

    def track_frame(

        self,

        frame: np.ndarray,

        persist: bool = True

    ) -> List[Detection]:

        """
        YOLO + ByteTrack pipeline.

        persist=True:
        keeps tracker state across frames.
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

                device=(

                    None

                    if CONFIG.DEVICE == "auto"

                    else CONFIG.DEVICE
                ),

                verbose=False
            )

        return self._parse_results(results)

    # ======================================
    # PARSE YOLO RESULTS
    # ======================================

    def _parse_results(
        self,
        results
    ) -> List[Detection]:

        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]

        boxes = getattr(
            result,
            "boxes",
            None
        )

        if boxes is None or len(boxes) == 0:
            return detections

        # ======================================
        # BOX DATA
        # ======================================

        xyxy = (

            boxes.xyxy.detach().cpu().numpy()

            if hasattr(boxes.xyxy, "detach")

            else np.asarray(boxes.xyxy)

        )

        confs = (

            boxes.conf.detach().cpu().numpy()

            if hasattr(boxes.conf, "detach")

            else np.asarray(boxes.conf)

        )

        clss = (

            boxes.cls.detach().cpu().numpy().astype(int)

            if hasattr(boxes.cls, "detach")

            else np.asarray(boxes.cls).astype(int)

        )

        # ======================================
        # TRACK IDS
        # ======================================

        ids = None

        if getattr(boxes, "id", None) is not None:

            ids = (

                boxes.id.detach().cpu().numpy().astype(int)

                if hasattr(boxes.id, "detach")

                else np.asarray(boxes.id).astype(int)

            )

        # ======================================
        # BUILD DETECTIONS
        # ======================================

        for i, bbox in enumerate(xyxy):

            cls_id = int(clss[i])

            class_name = self.names.get(
                cls_id,
                str(cls_id)
            )

            canonical = normalize_class_name(
                class_name
            )

            detections.append(

                Detection(

                    x1=float(bbox[0]),

                    y1=float(bbox[1]),

                    x2=float(bbox[2]),

                    y2=float(bbox[3]),

                    conf=float(confs[i]),

                    class_id=cls_id,

                    class_name=class_name,

                    canonical_class=canonical,

                    track_id=(
                        int(ids[i])
                        if ids is not None
                        else None
                    ),
                )
            )

        return detections


# ======================================
# DRAW DETECTIONS
# ======================================

def draw_detections(

    frame: np.ndarray,

    detections: Iterable[Detection],

    violations: Optional[List[Dict]] = None

) -> np.ndarray:

    """
    Draw:
    - bounding boxes
    - track IDs
    - ReID identities
    - similarity scores
    - violations
    """

    out = frame.copy()

    # ======================================
    # VIOLATION TRACK IDS
    # ======================================

    violation_track_ids = {

        str(v.get("track_id"))

        for v in (violations or [])

    }

    # ======================================
    # DRAW DETECTIONS
    # ======================================

    for det in detections:

        x1, y1, x2, y2 = map(
            int,
            det.bbox
        )

        # ======================================
        # VIOLATION STATUS
        # ======================================

        is_violation_person = (

            det.canonical_class
            ==
            CONFIG.PERSON_CLASS

            and

            str(det.track_id)
            in
            violation_track_ids
        )

        # ======================================
        # COLORS
        # ======================================

        color = (

            (0, 0, 255)

            if (
                is_violation_person
                or
                det.canonical_class.startswith("no_")
            )

            else

            (0, 180, 0)
        )

        if det.canonical_class == CONFIG.PERSON_CLASS:

            color = (

                (0, 0, 255)

                if is_violation_person

                else

                (255, 160, 0)
            )

        # ======================================
        # DRAW BBOX
        # ======================================

        cv2.rectangle(

            out,

            (x1, y1),

            (x2, y2),

            color,

            2
        )

        # ======================================
        # REID METADATA
        # ======================================

        global_id = (

            det.metadata.get(
                "reid_global_id",
                "unknown"
            )
        )

        similarity = (

            det.metadata.get(
                "reid_similarity",
                0.0
            )
        )

        # ======================================
        # MAIN LABEL
        # ======================================

        label = (
            f"{det.canonical_class} "
            f"{det.conf:.2f}"
        )

        if det.track_id is not None:

            label = (
                f"ID:{det.track_id} "
                f"{label}"
            )

        # ======================================
        # MAIN LABEL
        # ======================================

        cv2.putText(

            out,

            label,

            (x1, max(20, y1 - 7)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2
        )

        # ======================================
        # GLOBAL REID ID
        # ======================================

        cv2.putText(

            out,

            f"GID: {global_id}",

            (x1, max(45, y1 - 30)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 255),

            2
        )

        # ======================================
        # SIMILARITY SCORE
        # ======================================

        cv2.putText(

            out,

            f"SIM: {similarity:.2f}",

            (x1, max(70, y1 - 52)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 0),

            2
        )

    # ======================================
    # VIOLATION OVERLAY
    # ======================================

    if violations:

        y = 30

        for v in violations[:5]:

            msg = (

                f"VIOLATION: "

                f"ID {v.get('track_id')} "

                f"{v.get('violation_type')}"
            )

            cv2.putText(

                out,

                msg,

                (20, y),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2
            )

            y += 28

    return out