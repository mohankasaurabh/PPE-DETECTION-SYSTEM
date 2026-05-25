"""
# ======================================
# COMPLIANCE ENGINE / RULE ENGINE
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- Converts PPE associations into business-rule decisions.
- Separates:
    detection
        from
    safety policy.
- Supports:
    - per-camera PPE rules
    - future zone-based rules
    - identity-aware violations
    - enterprise analytics

Enterprise architecture:
- Detection models should NEVER contain business logic.
- Safety rules must remain configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .association_engine import PersonAssociation
from .config import CONFIG


# ======================================
# RULE SET
# ======================================

@dataclass
class ComplianceRuleSet:

    mandatory_ppe: List[str]

    optional_ppe: List[str]

    min_person_confidence: float = 0.25

    # ======================================
    # LOAD CAMERA RULES
    # ======================================

    @classmethod
    def from_camera_rules(

        cls,

        rules: Optional[Dict]

    ) -> "ComplianceRuleSet":

        rules = rules or {}

        return cls(

            mandatory_ppe=rules.get(

                "mandatory_ppe",

                CONFIG.DEFAULT_MANDATORY_PPE
            ),

            optional_ppe=rules.get(

                "optional_ppe",

                CONFIG.DEFAULT_OPTIONAL_PPE
            ),

            min_person_confidence=float(

                rules.get(
                    "min_person_confidence",
                    0.25
                )
            ),
        )


# ======================================
# COMPLIANCE ENGINE
# ======================================

class ComplianceEngine:

    # ======================================
    # INIT
    # ======================================

    def __init__(

        self,

        camera_rules: Optional[Dict] = None

    ):

        self.rules = (

            ComplianceRuleSet.from_camera_rules(
                camera_rules
            )
        )

    # ======================================
    # EVALUATE COMPLIANCE
    # ======================================

    def evaluate(

        self,

        associations: List[PersonAssociation]

    ) -> List[Dict]:

        violations: List[Dict] = []

        # ======================================
        # PROCESS WORKERS
        # ======================================

        for assoc in associations:

            person = assoc.person

            # ======================================
            # LOW CONFIDENCE FILTER
            # ======================================

            if (

                person.conf
                <
                self.rules.min_person_confidence

                or

                person.track_id is None
            ):

                continue

            # ======================================
            # REID IDENTITY
            # ======================================

            reid_global_id = (

                person.metadata.get(
                    "reid_global_id"
                )
            )

            # ======================================
            # PPE RULE CHECKS
            # ======================================

            for ppe in self.rules.mandatory_ppe:

                negative_class = (
                    f"no_{ppe}"
                )

                has_positive = (
                    assoc.has_ppe(ppe)
                )

                has_negative = (
                    assoc.has_negative(
                        negative_class
                    )
                )

                # ======================================
                # VIOLATION DETECTED
                # ======================================

                if (

                    has_negative

                    or

                    not has_positive
                ):

                    confidence = (

                        self._violation_confidence(

                            assoc,

                            ppe,

                            negative_class
                        )
                    )

                    # ======================================
                    # VIOLATION PAYLOAD
                    # ======================================

                    violation = {

                        # ======================================
                        # TRACKING
                        # ======================================

                        "track_id": (
                            person.track_id
                        ),

                        "reid_global_id": (
                            reid_global_id
                        ),

                        # ======================================
                        # CAMERA
                        # ======================================

                        "camera_id": None,

                        # ======================================
                        # VIOLATION
                        # ======================================

                        "violation_type": (
                            f"missing_{ppe}"
                        ),

                        "required_ppe": ppe,

                        "confidence": confidence,

                        # ======================================
                        # PERSON DATA
                        # ======================================

                        "person_bbox": list(
                            person.bbox
                        ),

                        "person_confidence": (
                            float(person.conf)
                        ),

                        # ======================================
                        # COMPLIANCE STATUS
                        # ======================================

                        "compliance": (
                            assoc.compliance_summary()
                        ),

                        # ======================================
                        # ASSOCIATION DATA
                        # ======================================

                        "association": (
                            assoc.as_dict()
                        ),

                        # ======================================
                        # VIOLATION TIMESTAMP
                        # ======================================

                        "timestamp": (
                            datetime.utcnow().isoformat()
                        ),

                        # ======================================
                        # ENTERPRISE SEVERITY
                        # ======================================

                        "severity": (
                            self._severity_level(
                                confidence
                            )
                        ),
                    }

                    violations.append(
                        violation
                    )

        return violations

    # ======================================
    # VIOLATION CONFIDENCE
    # ======================================

    def _violation_confidence(

        self,

        assoc: PersonAssociation,

        ppe: str,

        negative_class: str

    ) -> float:

        # ======================================
        # NEGATIVE PPE DETECTED
        # ======================================

        if assoc.has_negative(
            negative_class
        ):

            neg_conf = max([

                d.conf

                for d in assoc.negative_ppe[
                    negative_class
                ]

            ] or [0.75])

            return float(

                min(
                    0.98,
                    neg_conf + 0.10
                )
            )

        # ======================================
        # PPE ABSENCE
        # ======================================

        person_conf = (
            assoc.person.conf
        )

        assoc_penalty = (
            assoc.scores.get(
                ppe,
                0
            )
        )

        return float(

            max(

                0.45,

                min(

                    0.82,

                    person_conf
                    -
                    0.10
                    +
                    assoc_penalty * 0.2
                )
            )
        )

    # ======================================
    # SEVERITY LEVEL
    # ======================================

    @staticmethod
    def _severity_level(
        confidence: float
    ) -> str:

        if confidence >= 0.90:
            return "critical"

        if confidence >= 0.75:
            return "high"

        if confidence >= 0.60:
            return "medium"

        return "low"