"""
# ======================================
# ASSOCIATION ENGINE
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module decides which PPE item belongs to which tracked person.
- It solves the hard problem that YOLO detects objects independently: a helmet box and a vest box are not automatically attached to a worker.
- Enterprise PPE systems need this association layer to reduce false alerts in crowded scenes.

Algorithm summary:
- Split each person box into body regions: head, torso, hands, lower legs/feet.
- For each PPE object, score every candidate person using weighted features:
  1. Region IoU: does the PPE overlap the correct body region?
  2. Person overlap: is the PPE physically inside/overlapping the person box?
  3. Center distance: is the PPE center near the expected body region center?
  4. Detection confidence: how confident was the detector?
  5. Temporal consistency: was this PPE matched to this person in recent frames?
- Assign PPE to the highest scoring person above threshold.

Edge cases handled:
- Multiple people overlap: greedy score selects the most likely owner.
- Partial occlusion: distance and temporal consistency can still maintain association when IoU is weak.
- PPE false positives outside body area: filtered by minimum score.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from .config import CONFIG
from .detector import Detection

BBox = Tuple[float, float, float, float]


@dataclass
class PersonAssociation:
    person: Detection
    ppe: Dict[str, List[Detection]] = field(default_factory=lambda: defaultdict(list))
    negative_ppe: Dict[str, List[Detection]] = field(default_factory=lambda: defaultdict(list))
    scores: Dict[str, float] = field(default_factory=dict)

    def has_ppe(self, canonical_class: str) -> bool:
        return len(self.ppe.get(canonical_class, [])) > 0

    def has_negative(self, missing_class: str) -> bool:
        return len(self.negative_ppe.get(missing_class, [])) > 0

    def as_dict(self) -> Dict:
        return {
            "track_id": self.person.track_id,
            "person": self.person.as_dict(),
            "ppe": {k: [d.as_dict() for d in v] for k, v in self.ppe.items()},
            "negative_ppe": {k: [d.as_dict() for d in v] for k, v in self.negative_ppe.items()},
            "scores": self.scores,
        }


class AssociationEngine:
    BODY_REGION_BY_PPE = {
        "helmet": "head",
        "no_helmet": "head",
        "goggles": "head",
        "no_goggles": "head",
        "mask": "head",
        "no_mask": "head",
        "vest": "torso",
        "no_vest": "torso",
        "gloves": "hands",
        "no_gloves": "hands",
        "boots": "feet",
        "no_boots": "feet",
    }

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.temporal_cache: Dict[str, int] = {}
        self.history: Deque[Dict[str, int]] = deque(maxlen=30)

    def associate(self, detections: List[Detection]) -> List[PersonAssociation]:
        people = [d for d in detections if d.canonical_class == CONFIG.PERSON_CLASS]
        items = [d for d in detections if d.canonical_class in CONFIG.PPE_CLASSES or d.canonical_class in CONFIG.NEGATIVE_PPE_CLASSES]
        associations = {p.track_id: PersonAssociation(person=p) for p in people if p.track_id is not None}
        if not people or not items:
            return list(associations.values())

        frame_assignments: Dict[str, int] = {}
        for item in items:
            best_person: Optional[Detection] = None
            best_score = 0.0
            for person in people:
                score = self._score_pair(person, item)
                if score > best_score:
                    best_score = score
                    best_person = person
            if best_person is not None and best_person.track_id is not None and best_score >= CONFIG.ASSOCIATION_MIN_SCORE:
                pa = associations.get(best_person.track_id)
                if pa is None:
                    pa = PersonAssociation(best_person)
                    associations[best_person.track_id] = pa
                if item.canonical_class.startswith("no_"):
                    pa.negative_ppe[item.canonical_class].append(item)
                else:
                    pa.ppe[item.canonical_class].append(item)
                pa.scores[item.canonical_class] = max(pa.scores.get(item.canonical_class, 0), round(best_score, 4))
                cache_key = self._cache_key(item)
                frame_assignments[cache_key] = best_person.track_id
                self.temporal_cache[cache_key] = best_person.track_id

        self.history.append(frame_assignments)
        return list(associations.values())

    def _score_pair(self, person: Detection, item: Detection) -> float:
        region_name = self.BODY_REGION_BY_PPE.get(item.canonical_class, "torso")
        region_box = self.body_region(person.bbox, region_name)
        region_iou = iou(region_box, item.bbox)
        person_overlap = intersection_area(person.bbox, item.bbox) / max(item.area, 1.0)
        distance_score = normalized_center_distance_score(region_box, item.bbox)
        temporal_bonus = CONFIG.TEMPORAL_ASSOC_BONUS if self.temporal_cache.get(self._cache_key(item)) == person.track_id else 0.0

        score = (
            0.42 * region_iou +
            0.22 * person_overlap +
            0.18 * distance_score +
            0.10 * item.conf +
            temporal_bonus
        )
        return float(min(score, 1.0))

    def _cache_key(self, item: Detection) -> str:
        if item.track_id is not None:
            return f"{self.camera_id}:{item.canonical_class}:tid:{item.track_id}"
        cx, cy = item.center
        return f"{self.camera_id}:{item.canonical_class}:grid:{int(cx//32)}:{int(cy//32)}"

    @staticmethod
    def body_region(person_box: BBox, region: str) -> BBox:
        x1, y1, x2, y2 = person_box
        w, h = x2 - x1, y2 - y1
        if region == "head":
            return x1 + 0.18 * w, y1, x2 - 0.18 * w, y1 + 0.28 * h
        if region == "torso":
            return x1 + 0.08 * w, y1 + 0.22 * h, x2 - 0.08 * w, y1 + 0.72 * h
        if region == "hands":
            return x1 - 0.05 * w, y1 + 0.25 * h, x2 + 0.05 * w, y1 + 0.78 * h
        if region == "feet":
            return x1 + 0.05 * w, y1 + 0.68 * h, x2 - 0.05 * w, y2
        return person_box


def intersection_area(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def normalized_center_distance_score(region: BBox, item: BBox) -> float:
    rx = (region[0] + region[2]) / 2
    ry = (region[1] + region[3]) / 2
    ix = (item[0] + item[2]) / 2
    iy = (item[1] + item[3]) / 2
    rw = max(1.0, region[2] - region[0])
    rh = max(1.0, region[3] - region[1])
    norm_dist = (((rx - ix) / rw) ** 2 + ((ry - iy) / rh) ** 2) ** 0.5
    return float(max(0.0, 1.0 - norm_dist))
