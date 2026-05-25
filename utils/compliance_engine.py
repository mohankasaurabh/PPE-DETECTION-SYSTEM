"""
# ======================================
# COMPLIANCE ENGINE / RULE ENGINE
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This module converts detection association results into business decisions: compliant or non-compliant.
- It supports per-camera and future per-zone mandatory PPE rules.
- Enterprise systems separate rule evaluation from detection so safety policies can change without retraining YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .association_engine import PersonAssociation
from .config import CONFIG


@dataclass
class ComplianceRuleSet:
    mandatory_ppe: List[str]
    optional_ppe: List[str]
    min_person_confidence: float = 0.25

    @classmethod
    def from_camera_rules(cls, rules: Optional[Dict]) -> "ComplianceRuleSet":
        rules = rules or {}
        return cls(
            mandatory_ppe=rules.get("mandatory_ppe", CONFIG.DEFAULT_MANDATORY_PPE),
            optional_ppe=rules.get("optional_ppe", CONFIG.DEFAULT_OPTIONAL_PPE),
            min_person_confidence=float(rules.get("min_person_confidence", 0.25)),
        )


class ComplianceEngine:
    def __init__(self, camera_rules: Optional[Dict] = None):
        self.rules = ComplianceRuleSet.from_camera_rules(camera_rules)

    def evaluate(self, associations: List[PersonAssociation]) -> List[Dict]:
        violations: List[Dict] = []
        for assoc in associations:
            person = assoc.person
            if person.conf < self.rules.min_person_confidence or person.track_id is None:
                continue

            for ppe in self.rules.mandatory_ppe:
                negative_class = f"no_{ppe}"
                has_positive = assoc.has_ppe(ppe)
                has_negative = assoc.has_negative(negative_class)

                if has_negative or not has_positive:
                    confidence = self._violation_confidence(assoc, ppe, negative_class)
                    violations.append({
                        "track_id": person.track_id,
                        "reid_global_id": person.metadata.get("reid_global_id"),
                        "camera_id": None,
                        "violation_type": f"missing_{ppe}",
                        "required_ppe": ppe,
                        "confidence": confidence,
                        "person_bbox": list(person.bbox),
                        "association": assoc.as_dict(),
                    })
        return violations

    def _violation_confidence(self, assoc: PersonAssociation, ppe: str, negative_class: str) -> float:
        if assoc.has_negative(negative_class):
            neg_conf = max([d.conf for d in assoc.negative_ppe[negative_class]] or [0.75])
            return float(min(0.98, neg_conf + 0.10))
        # Missing by absence has lower confidence because detection may be occluded.
        person_conf = assoc.person.conf
        assoc_penalty = assoc.scores.get(ppe, 0)
        return float(max(0.45, min(0.82, person_conf - 0.10 + assoc_penalty * 0.2)))
