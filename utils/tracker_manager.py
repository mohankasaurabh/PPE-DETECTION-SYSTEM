"""
# ======================================
# TRACKER MANAGER
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module keeps track-level memory above raw ByteTrack IDs.
- ByteTrack gives per-stream object IDs, but enterprise alerting needs track age, last seen time, missing-frame counters, FPS-independent duration, and recovery hooks.
- This layer becomes the bridge between low-level tracking and higher-level event lifecycle logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional, Tuple

from .config import CONFIG
from .detector import Detection


@dataclass
class TrackState:
    track_id: int
    camera_id: str
    bbox: Tuple[float, float, float, float]
    first_seen: float = field(default_factory=time)
    last_seen: float = field(default_factory=time)
    age_frames: int = 0
    missed_frames: int = 0
    canonical_class: str = CONFIG.PERSON_CLASS
    reid_global_id: Optional[str] = None

    def update(self, det: Detection) -> None:
        self.bbox = det.bbox
        self.last_seen = time()
        self.age_frames += 1
        self.missed_frames = 0


class TrackerManager:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.tracks: Dict[int, TrackState] = {}
        self._next_fallback_id = 1_000_000

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Ensure every person detection has a stable track ID and update track memory."""
        seen_ids = set()
        for det in detections:
            if det.canonical_class != CONFIG.PERSON_CLASS:
                continue
            if det.track_id is None:
                det.track_id = self._assign_fallback_id(det)
            seen_ids.add(det.track_id)
            state = self.tracks.get(det.track_id)
            if state is None:
                self.tracks[det.track_id] = TrackState(
                    track_id=det.track_id, camera_id=self.camera_id, bbox=det.bbox, canonical_class=det.canonical_class
                )
            else:
                state.update(det)

        stale = []
        for tid, state in self.tracks.items():
            if tid not in seen_ids:
                state.missed_frames += 1
                if time() - state.last_seen > CONFIG.REID_MAX_AGE_SECONDS * 2:
                    stale.append(tid)
        for tid in stale:
            self.tracks.pop(tid, None)
        return detections

    def _assign_fallback_id(self, det: Detection) -> int:
        # If tracker output is missing, assign an approximate ID by matching center to existing recent tracks.
        cx, cy = det.center
        best_id = None
        best_dist = float("inf")
        for tid, state in self.tracks.items():
            sx = (state.bbox[0] + state.bbox[2]) / 2
            sy = (state.bbox[1] + state.bbox[3]) / 2
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if dist < best_dist and dist < 60:
                best_id = tid
                best_dist = dist
        if best_id is not None:
            return best_id
        self._next_fallback_id += 1
        return self._next_fallback_id
