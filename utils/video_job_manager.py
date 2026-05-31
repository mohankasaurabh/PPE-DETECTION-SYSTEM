"""
# ======================================
# VIDEO JOB MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Processes an uploaded video asynchronously in a background worker thread
  using the same detection / tracking / ReID / association / compliance
  pipeline as live streams.
- While the video is being analysed it exposes the latest annotated frame as
  an MJPEG stream so the browser can watch detection happen "live", side by
  side with real-time PPE violation alerts.
- When processing finishes it stores the rendered output video plus an
  aggregated, whole-video PPE compliance summary so the final result page can
  replay the annotated footage and show per-worker compliance.

Why this architecture:
- The previous implementation processed the upload synchronously inside the
  request, so the user saw nothing until the whole clip finished and the final
  page had no compliance summary. Moving the work to a tracked background job
  unlocks the live view and a rich result page without changing the core CV
  modules.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np

from .association_engine import AssociationEngine
from .compliance_engine import ComplianceEngine
from .config import CONFIG
from .detector import PPEDetector, draw_detections
from .event_manager import EventManager
from .logger import get_logger
from .reid_manager import ReIDManager
from .tracker_manager import TrackerManager

logger = get_logger("ppe.video_job")

REQUIRED_PPE = ["helmet", "vest"]


# ======================================
# SINGLE VIDEO JOB
# ======================================

class VideoJob:

    def __init__(self, job_id: str, input_path: Path):
        self.job_id = job_id
        self.input_path = Path(input_path)

        # lifecycle
        self.state = "processing"  # processing | done | error
        self.error: Optional[str] = None
        self.started_at = datetime.utcnow().isoformat()

        # progress
        self.total_frames = 0
        self.processed_frames = 0
        self.violation_count = 0

        # results
        self.output_path: Optional[str] = None     # web path to final video
        self.persons_summary: List[Dict] = []
        self.alerts: List[Dict] = []               # newest first

        # live frame buffer
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()

        # aggregation state
        self._person_state: Dict[str, Dict] = {}
        self._alert_keys: set = set()

    # ------------------------------------
    # LIVE FRAME
    # ------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, CONFIG.JPEG_QUALITY],
        )
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    # ------------------------------------
    # STATUS SNAPSHOT
    # ------------------------------------

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return round(
            min(100.0, (self.processed_frames / self.total_frames) * 100.0),
            1,
        )

    def status(self) -> Dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "error": self.error,
            "progress": self.progress,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "violation_count": self.violation_count,
            "alerts": self.alerts[:12],
            "result_url": (
                f"/video/result/{self.job_id}"
                if self.state == "done"
                else None
            ),
        }

    def result(self) -> Dict:
        return {
            "job_id": self.job_id,
            "output_path": self.output_path,
            "frames": self.total_frames,
            "violation_count": self.violation_count,
            "persons_summary": self.persons_summary,
        }


# ======================================
# JOB MANAGER (SINGLETON)
# ======================================

class VideoJobManager:

    def __init__(self):
        self.jobs: Dict[str, VideoJob] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self.jobs.get(job_id)

    # ------------------------------------
    # START A JOB
    # ------------------------------------

    def start(self, input_path: Path, rules: Optional[Dict] = None) -> VideoJob:
        job_id = "video_" + uuid.uuid4().hex[:10]
        job = VideoJob(job_id, input_path)

        with self._lock:
            self.jobs[job_id] = job

        worker = threading.Thread(
            target=self._run,
            args=(job, rules or {}),
            daemon=True,
        )
        worker.start()

        return job

    # ------------------------------------
    # WORKER LOOP
    # ------------------------------------

    def _run(self, job: VideoJob, rules: Dict) -> None:
        try:
            self._process(job, rules)
            job.persons_summary = self._build_summary(job)
            job.state = "done"
            logger.info(
                "Video job %s finished (%s violations, %s persons)",
                job.job_id,
                job.violation_count,
                len(job.persons_summary),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Video job %s failed: %s", job.job_id, exc)
            job.error = str(exc)
            job.state = "error"

    def _process(self, job: VideoJob, rules: Dict) -> None:
        output_path = CONFIG.OUTPUT_VIDEO_DIR / f"{job.job_id}.mp4"

        cap = cv2.VideoCapture(str(job.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {job.input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = self._open_writer(output_path, fps, w, h)

        detector = PPEDetector()
        tracker = TrackerManager(job.job_id)
        reid = ReIDManager(job.job_id)
        association = AssociationEngine(job.job_id)
        compliance = ComplianceEngine(rules or {})
        events = EventManager(job.job_id, "video")

        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_id += 1
                job.processed_frames = frame_id

                if frame_id % max(1, CONFIG.DEFAULT_FRAME_SKIP) == 0:
                    detections = detector.track_frame(frame, persist=True)
                    detections = tracker.update(frame, detections)

                    people = [
                        d for d in detections
                        if d.canonical_class == CONFIG.PERSON_CLASS
                    ]
                    reid.update_person_identities(frame, people)

                    associations = association.associate(detections)
                    violations = compliance.evaluate(associations)
                    for v in violations:
                        v["camera_id"] = job.job_id

                    job.violation_count += len(violations)

                    annotated = draw_detections(frame, detections, violations)
                    events.update(violations, annotated, frame)

                    self._register_persons(job, associations)
                    self._register_alerts(job, violations, frame)

                    writer.write(annotated)
                    job.set_frame(annotated)
                else:
                    writer.write(frame)
                    # Keep the live view smooth on skipped frames too.
                    if frame_id % 2 == 0:
                        job.set_frame(frame)
        finally:
            cap.release()
            writer.release()

        if job.total_frames <= 0:
            job.total_frames = frame_id

        job.output_path = f"/static/outputs/videos/{output_path.name}"

    # ------------------------------------
    # VIDEO WRITER (browser-friendly codec)
    # ------------------------------------

    def _open_writer(self, output_path: Path, fps, w, h) -> cv2.VideoWriter:
        # Prefer H.264 (avc1) so the result plays in Chrome; fall back to mp4v.
        for fourcc_name in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if writer.isOpened():
                logger.info("Video writer using codec %s", fourcc_name)
                return writer
            writer.release()
        raise RuntimeError("Could not open a video writer for the output file.")

    # ------------------------------------
    # PERSON AGGREGATION
    # ------------------------------------

    def _person_key(self, track_id, reid_global_id) -> str:
        if reid_global_id:
            return str(reid_global_id)
        return f"track_{track_id}"

    def _register_persons(self, job: VideoJob, associations) -> None:
        for assoc in associations:
            data = assoc.as_dict()
            reid_global_id = data.get("reid_global_id")
            track_id = data.get("track_id")
            key = self._person_key(track_id, reid_global_id)

            state = job._person_state.setdefault(
                key,
                {
                    "display_id": reid_global_id or track_id,
                    "missing": set(),
                    "order": len(job._person_state),
                },
            )

            ppe = data.get("ppe", {}) or {}
            for item in REQUIRED_PPE:
                has_item = item in ppe and len(ppe[item]) > 0
                if not has_item:
                    state["missing"].add(item)

    def _build_summary(self, job: VideoJob) -> List[Dict]:
        persons = []
        ordered = sorted(
            job._person_state.items(),
            key=lambda kv: kv[1]["order"],
        )
        for index, (_key, state) in enumerate(ordered):
            items = {}
            violations = []
            for item in REQUIRED_PPE:
                has_item = item not in state["missing"]
                items[item] = has_item
                if not has_item:
                    violations.append(f"{item.title()} Missing")

            persons.append(
                {
                    "person_number": index + 1,
                    "track_id": state["display_id"],
                    "items": items,
                    "violations": violations,
                    "compliant": len(violations) == 0,
                }
            )
        return persons

    # ------------------------------------
    # LIVE ALERTS
    # ------------------------------------

    def _register_alerts(self, job: VideoJob, violations, frame) -> None:
        for v in violations:
            reid_global_id = v.get("reid_global_id")
            track_id = v.get("track_id")
            key = (
                self._person_key(track_id, reid_global_id),
                v.get("violation_type"),
            )
            if key in job._alert_keys:
                continue
            job._alert_keys.add(key)

            snapshot = self._save_crop(
                job,
                frame,
                v.get("person_bbox"),
                len(job._alert_keys),
            )

            job.alerts.insert(
                0,
                {
                    "track_id": reid_global_id or track_id,
                    "violation_type": (
                        v.get("violation_type", "PPE Violation")
                        .replace("_", " ")
                        .title()
                    ),
                    "created_at": v.get("timestamp", ""),
                    "snapshot": snapshot,
                },
            )

    def _save_crop(self, job, frame, bbox, index) -> Optional[str]:
        if not bbox:
            return None
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            out_dir = CONFIG.VIOLATION_DIR / "video"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{job.job_id}_alert_{index}.jpg"
            cv2.imwrite(str(out_dir / fname), crop)
            return f"/static/violations/video/{fname}"
        except Exception:  # noqa: BLE001
            return None


# ======================================
# SINGLETON INSTANCE
# ======================================

VIDEO_JOBS = VideoJobManager()


def frame_generator(job_id: str) -> Generator[bytes, None, None]:
    """MJPEG generator that streams the latest annotated frame of a job."""

    boundary_open = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"

    placeholder = _placeholder_jpeg()

    while True:
        job = VIDEO_JOBS.get(job_id)
        if job is None:
            break

        frame = job.get_frame() or placeholder
        yield boundary_open + frame + b"\r\n"

        if job.state != "processing":
            # Send the final frame once more, then stop so the <img> holds it.
            final = job.get_frame() or placeholder
            yield boundary_open + final + b"\r\n"
            break

        time.sleep(0.08)


def _placeholder_jpeg() -> bytes:
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (17, 24, 39)
    cv2.putText(
        canvas,
        "Preparing video...",
        (130, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (148, 163, 184),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", canvas)
    return buf.tobytes() if ok else b""
