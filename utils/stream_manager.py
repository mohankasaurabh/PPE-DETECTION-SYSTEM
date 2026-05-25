"""
# ======================================
# STREAM MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module owns multi-camera ingestion, processing, health monitoring, and MJPEG streaming.
- It solves the enterprise requirement that each camera must run independently with reconnects and its own processing state.
- The architecture uses separate acquisition and processing threads per source, connected through FrameBuffer.

Important design choices:
- Each camera worker owns a PPEDetector instance so ByteTrack state remains per-camera.
- Frame acquisition keeps reading even when inference is slower; the buffer drops stale frames when needed.
- Processing thread performs YOLO, tracking, ReID, association, rule evaluation, event lifecycle, evidence saving, and dashboard emission.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np

from .alert_manager import ALERTS
from .analytics_engine import ANALYTICS
from .association_engine import AssociationEngine
from .compliance_engine import ComplianceEngine
from .config import CONFIG
from .database import DB
from .detector import PPEDetector, draw_detections
from .event_manager import EventManager
from .frame_buffer import FrameBuffer, FramePacket
from .logger import get_logger
from .reid_manager import ReIDManager
from .stats import RuntimeStats
from .tracker_manager import TrackerManager

logger = get_logger("ppe.stream_manager")


class CameraWorker:
    def __init__(self, camera: Dict):
        self.camera = camera
        self.camera_id = camera["camera_id"]
        self.source_type = camera.get("source_type", "rtsp")
        self.source_uri = camera.get("source_uri", "0")
        self.rules = camera.get("rules", {}) or {}
        self.stop_event = threading.Event()
        self.frame_buffer = FrameBuffer()
        self.stats = RuntimeStats(camera_id=self.camera_id)
        self.latest_jpeg: Optional[bytes] = None
        self.latest_frame_lock = threading.RLock()
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None

        self.detector = PPEDetector()
        self.tracker = TrackerManager(self.camera_id)
        self.reid = ReIDManager(self.camera_id)
        self.association = AssociationEngine(self.camera_id)
        self.compliance = ComplianceEngine(self.rules)
        self.events = EventManager(self.camera_id, "webcam" if self.source_type == "webcam" else "rtsp")

    def start(self) -> None:
        if self.capture_thread and self.capture_thread.is_alive():
            return
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, name=f"capture-{self.camera_id}", daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, name=f"process-{self.camera_id}", daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
        DB.update_camera_status(self.camera_id, "STARTING")
        ALERTS.emit_camera_status(self.camera_id, "STARTING")

    def stop(self) -> None:
        self.stop_event.set()
        self.frame_buffer.clear()
        DB.update_camera_status(self.camera_id, "STOPPED")
        ALERTS.emit_camera_status(self.camera_id, "STOPPED")

    def _open_capture(self):
        source = int(self.source_uri) if self.source_type == "webcam" and str(self.source_uri).isdigit() else self.source_uri
        cap = cv2.VideoCapture(source)
        if self.source_type == "rtsp":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _capture_loop(self) -> None:
        frame_id = 0
        cap = None
        while not self.stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = self._open_capture()
                    if not cap.isOpened():
                        self.stats.last_error = "Unable to open source"
                        DB.update_camera_status(self.camera_id, "DISCONNECTED")
                        ALERTS.emit_camera_status(self.camera_id, "DISCONNECTED", {"error": self.stats.last_error})
                        time.sleep(CONFIG.CAPTURE_RECONNECT_SECONDS)
                        continue
                    DB.update_camera_status(self.camera_id, "RUNNING")
                    ALERTS.emit_camera_status(self.camera_id, "RUNNING")

                ok, frame = cap.read()
                if not ok or frame is None:
                    self.stats.last_error = "Frame read failed"
                    if cap:
                        cap.release()
                    cap = None
                    time.sleep(CONFIG.CAPTURE_RECONNECT_SECONDS)
                    continue

                frame_id += 1
                self.stats.frames_read += 1
                self.stats.last_frame_ts = time.time()
                self.frame_buffer.put(FramePacket(frame=frame, frame_id=frame_id, timestamp=time.time()))
                self.stats.dropped_frames = self.frame_buffer.dropped_frames
            except Exception as exc:
                self.stats.last_error = str(exc)
                logger.exception("Capture loop failed for %s: %s", self.camera_id, exc)
                time.sleep(CONFIG.CAPTURE_RECONNECT_SECONDS)
        if cap:
            cap.release()

    def _process_loop(self) -> None:
        while not self.stop_event.is_set():
            packet = self.frame_buffer.get(timeout=0.5)
            if packet is None:
                continue
            if packet.frame_id % max(1, CONFIG.DEFAULT_FRAME_SKIP) != 0:
                continue
            try:
                raw_frame = packet.frame
                detections = self.detector.track_frame(raw_frame, persist=True)
                detections = self.tracker.update(detections)
                people = [d for d in detections if d.canonical_class == CONFIG.PERSON_CLASS]
                self.reid.update_person_identities(raw_frame, people)
                associations = self.association.associate(detections)
                violations = self.compliance.evaluate(associations)
                for v in violations:
                    v["camera_id"] = self.camera_id
                annotated = draw_detections(raw_frame, detections, violations)
                event_payloads = self.events.update(violations, annotated, raw_frame)
                self._update_latest_frame(annotated)
                self.stats.frames_processed += 1
                self.stats.last_process_ts = time.time()

                ALERTS.emit_events(event_payloads)
                if self.stats.frames_processed % 15 == 0:
                    ALERTS.emit_metrics({
                        "camera_id": self.camera_id,
                        "stats": self.stats.as_dict(),
                        "analytics": ANALYTICS.summary(),
                    })
            except Exception as exc:
                self.stats.last_error = str(exc)
                logger.exception("Processing loop failed for %s: %s", self.camera_id, exc)

    def _update_latest_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.JPEG_QUALITY])
        if ok:
            with self.latest_frame_lock:
                self.latest_jpeg = encoded.tobytes()
                self.stats.frames_streamed += 1

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.latest_frame_lock:
            return self.latest_jpeg


class StreamManager:
    def __init__(self):
        self.workers: Dict[str, CameraWorker] = {}
        self._lock = threading.RLock()

    def start_camera(self, camera_id: str) -> bool:
        camera = DB.get_camera(camera_id)
        if not camera:
            return False
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker is None:
                worker = CameraWorker(camera)
                self.workers[camera_id] = worker
            worker.start()
        return True

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            worker = self.workers.get(camera_id)
            if not worker:
                DB.update_camera_status(camera_id, "STOPPED")
                return False
            worker.stop()
        return True

    def restart_camera(self, camera_id: str) -> bool:
        self.stop_camera(camera_id)
        time.sleep(0.5)
        return self.start_camera(camera_id)

    def stop_all(self) -> None:
        with self._lock:
            for worker in self.workers.values():
                worker.stop()

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        return self.workers.get(camera_id)

    def list_runtime_stats(self) -> List[Dict]:
        with self._lock:
            return [w.stats.as_dict() for w in self.workers.values()]

    def frame_generator(self, camera_id: str) -> Generator[bytes, None, None]:
        while True:
            worker = self.get_worker(camera_id)
            frame = worker.get_latest_jpeg() if worker else None
            if frame is None:
                frame = self._placeholder_frame(camera_id)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1 / max(1, CONFIG.TARGET_STREAM_FPS))

    @staticmethod
    def _placeholder_frame(camera_id: str) -> bytes:
        canvas = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(canvas, f"Waiting for camera {camera_id}", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        ok, encoded = cv2.imencode(".jpg", canvas)
        return encoded.tobytes() if ok else b""


STREAMS = StreamManager()
