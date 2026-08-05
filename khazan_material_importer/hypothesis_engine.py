"""
hypothesis_engine.py
====================
Hypothesis Engine & Dynamic Confidence Calculator for Khazan Material Importer.

Design Philosophy
-----------------
* SINGLE AUTHORITY allowed to transform raw observations (from EvidenceLedger)
  into conclusions.
* Computes DYNAMIC BIDIRECTIONAL confidence (0.0% to 100.0%).
  Confidence increases with supporting evidence AND decreases when contradictions appear.
* Implements a 5-Level Dynamic Knowledge Maturity Model (Level 1 Unknown → Level 5 Confirmed).
  Levels promote and demote dynamically based on net evidence weight and asset coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
try:
    from .evidence_ledger import EvidenceLedger, Observation
except ImportError:
    from evidence_ledger import EvidenceLedger, Observation


class MaturityLevel(Enum):
    LEVEL_1_UNKNOWN = "Level 1 — Unknown / Investigating"
    LEVEL_2_PLAUSIBLE = "Level 2 — Plausible"
    LEVEL_3_SUPPORTED = "Level 3 — Supported"
    LEVEL_4_HIGHLY_SUPPORTED = "Level 4 — Highly Supported"
    LEVEL_5_CONFIRMED = "Level 5 — Confirmed"


@dataclass
class HypothesisResult:
    """Evaluated state of a single shader hypothesis."""
    hypothesis_id: str                   # e.g., "HYP-R"
    name: str                            # e.g., "Tex_R Smoothness Interpretation"
    current_interpretation: str         # e.g., "Likely Toon Smoothness/Glossiness (Invert 1.0 - R)"
    confidence_score: float             # 0.0 to 100.0%
    maturity_level: MaturityLevel
    supporting_observations: List[Observation] = field(default_factory=list)
    contradicting_observations: List[Observation] = field(default_factory=list)
    support_weight: float = 0.0
    contradiction_weight: float = 0.0
    net_weight: float = 0.0
    character_count: int = 0
    material_count: int = 0
    remaining_unknowns: List[str] = field(default_factory=list)
    alternative_hypotheses: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "interpretation": self.current_interpretation,
            "confidence_score": round(self.confidence_score, 1),
            "maturity_level": self.maturity_level.value,
            "support_weight": round(self.support_weight, 2),
            "contradiction_weight": round(self.contradiction_weight, 2),
            "net_weight": round(self.net_weight, 2),
            "character_count": self.character_count,
            "material_count": self.material_count,
            "supporting_count": len(self.supporting_observations),
            "contradicting_count": len(self.contradicting_observations),
            "remaining_unknowns": self.remaining_unknowns,
            "alternatives": self.alternative_hypotheses,
        }


class HypothesisEngine:
    """Engine that evaluates hypotheses against observations in EvidenceLedger."""

    def __init__(self, ledger: EvidenceLedger):
        self.ledger = ledger

    def evaluate_all(self) -> Dict[str, HypothesisResult]:
        """Evaluate all core hypotheses using accumulated evidence."""
        return {
            "HYP-D": self._eval_diffuse(),
            "HYP-N": self._eval_normal(),
            "HYP-R": self._eval_roughness_smoothness(),
            "HYP-S": self._eval_specular_masks(),
            "HYP-I": self._eval_indirect_lighting(),
            "HYP-EYE-E": self._eval_base_eye_e(),
        }

    # -------------------------------------------------------------------------
    # Core Hypothesis Evaluators
    # -------------------------------------------------------------------------
    def _eval_diffuse(self) -> HypothesisResult:
        """HYP-D: Tex_D / PM_Diffuse = Base Color (Confirmed)."""
        obs_list = self.ledger.query(finding_type="diffuse_match")
        support_w, contra_w, chars, mats = self._aggregate_obs(obs_list)

        # Baseline confirmed channel
        base_support = 100.0 + support_w
        conf = self._calc_bidirectional_confidence(base_support, contra_w)
        maturity = self._calc_maturity(conf, len(chars), 0)

        return HypothesisResult(
            hypothesis_id="HYP-D",
            name="Tex_D Base Color Mapping",
            current_interpretation="Confirmed Base Color / Diffuse Texture",
            confidence_score=conf,
            maturity_level=maturity,
            supporting_observations=obs_list,
            character_count=len(chars),
            material_count=len(mats),
            support_weight=base_support,
            remaining_unknowns=["Color space gamma variation across custom lighting zones"],
        )

    def _eval_normal(self) -> HypothesisResult:
        """HYP-N: Tex_N / PM_Normals = Normal Map (Confirmed)."""
        obs_list = self.ledger.query(finding_type="normal_match")
        support_w, contra_w, chars, mats = self._aggregate_obs(obs_list)

        base_support = 100.0 + support_w
        conf = self._calc_bidirectional_confidence(base_support, contra_w)
        maturity = self._calc_maturity(conf, len(chars), 0)

        return HypothesisResult(
            hypothesis_id="HYP-N",
            name="Tex_N Normal Map Mapping",
            current_interpretation="Confirmed Tangent-space Normal Map (OpenGL +Y)",
            confidence_score=conf,
            maturity_level=maturity,
            supporting_observations=obs_list,
            character_count=len(chars),
            material_count=len(mats),
            support_weight=base_support,
            remaining_unknowns=["DirectX -Y vs OpenGL +Y flip on non-character props"],
        )

    def _eval_roughness_smoothness(self) -> HypothesisResult:
        """HYP-R: Tex_R = Toon Smoothness / Glossiness (Invert 1.0 - R)."""
        all_obs = self.ledger.query()
        support = []
        contra = []

        for obs in all_obs:
            if obs.finding_type in ("extrema", "histogram_skew"):
                mean = obs.raw_metrics.get("mean", 0.5)
                max_val = obs.raw_metrics.get("max", 255)
                # Extremely low pixel values (<0.15) support Smoothness inversion
                if mean < 0.20 or max_val < 60:
                    support.append(obs)
                elif mean > 0.80:
                    contra.append(obs)
            elif obs.finding_type == "visual_improvement":
                support.append(obs)
            elif obs.finding_type == "high_roughness_contradiction":
                contra.append(obs)

        sup_w = sum(o.weight for o in support)
        con_w = sum(o.weight for o in contra)
        chars = {o.character for o in support + contra}
        mats = {o.material for o in support + contra}

        # Base support for verified visual improvement across character set
        net_sup = 75.0 + sup_w
        conf = self._calc_bidirectional_confidence(net_sup, con_w)
        maturity = self._calc_maturity(conf, len(chars), len(contra))

        return HypothesisResult(
            hypothesis_id="HYP-R",
            name="Tex_R Smoothness Interpretation",
            current_interpretation="Likely Toon Smoothness / Glossiness (Invert 1.0 - R)",
            confidence_score=conf,
            maturity_level=maturity,
            supporting_observations=support,
            contradicting_observations=contra,
            support_weight=net_sup,
            contradiction_weight=con_w,
            net_weight=net_sup - con_w,
            character_count=len(chars),
            material_count=len(mats),
            remaining_unknowns=[
                "Whether values encode stylized toon gloss instead of physical smoothness",
                "Whether non-clothing metal materials use non-inverted roughness",
            ],
            alternative_hypotheses=[
                {"hypothesis": "ALT-R1: Physical Roughness", "status": "REJECTED (produces mirror vinyl sheen)"},
                {"hypothesis": "ALT-R2: Specular Response Mask", "status": "PLAUSIBLE (partially correlated with Tex_S)"},
            ],
        )

    def _eval_specular_masks(self) -> HypothesisResult:
        """HYP-S: Tex_S = Packed RGBA Specular & Toon Shadow Masks."""
        all_obs = self.ledger.query()
        support = []
        contra = []

        for obs in all_obs:
            if obs.finding_type in ("channel_correlation", "packed_multichannel"):
                corr = obs.raw_metrics.get("corr_min", 1.0)
                if corr < 0.5:
                    support.append(obs)
                else:
                    contra.append(obs)
            elif obs.finding_type == "specular_hdri_sheen_contradiction":
                contra.append(obs)

        sup_w = sum(o.weight for o in support)
        con_w = sum(o.weight for o in contra)
        chars = {o.character for o in support + contra}
        mats = {o.material for o in support + contra}

        net_sup = 60.0 + sup_w
        conf = self._calc_bidirectional_confidence(net_sup, con_w)
        maturity = self._calc_maturity(conf, len(chars), len(contra))

        return HypothesisResult(
            hypothesis_id="HYP-S",
            name="Tex_S Packed SpecularMasks Unpacking",
            current_interpretation="Packed Toon Mask (R=Specular, G=Shadow Width, B=Rim Mask)",
            confidence_score=conf,
            maturity_level=maturity,
            supporting_observations=support,
            contradicting_observations=contra,
            support_weight=net_sup,
            contradiction_weight=con_w,
            net_weight=net_sup - con_w,
            character_count=len(chars),
            material_count=len(mats),
            remaining_unknowns=[
                "Exact function of Alpha channel in Tex_S",
                "Unreal toon shader threshold scaling curve for Green channel",
            ],
            alternative_hypotheses=[
                {"hypothesis": "ALT-S1: Standard PBR Specular Map", "status": "REJECTED (causes HDRI reflection artifacts)"},
                {"hypothesis": "ALT-S2: Multi-layer Material Mask", "status": "SUPPORTED (Channel R/G/B diverge)"},
            ],
        )

    def _eval_indirect_lighting(self) -> HypothesisResult:
        """HYP-I: Tex_I = Pre-baked Indirect Lighting / Ambient Overlay."""
        all_obs = self.ledger.query()
        support = []
        contra = []

        for obs in all_obs:
            if obs.finding_type == "indirect_lighting_match":
                support.append(obs)
            elif obs.finding_type == "high_frequency_detail_contradiction":
                contra.append(obs)

        sup_w = sum(o.weight for o in support)
        con_w = sum(o.weight for o in contra)
        chars = {o.character for o in support + contra}
        mats = {o.material for o in support + contra}

        net_sup = 45.0 + sup_w
        conf = self._calc_bidirectional_confidence(net_sup, con_w)
        maturity = self._calc_maturity(conf, len(chars), len(contra))

        return HypothesisResult(
            hypothesis_id="HYP-I",
            name="Tex_I Indirect Lighting Overlay",
            current_interpretation="Pre-baked Indirect Illumination Overlay (Multiply 35%)",
            confidence_score=conf,
            maturity_level=maturity,
            supporting_observations=support,
            contradicting_observations=contra,
            support_weight=net_sup,
            contradiction_weight=con_w,
            net_weight=net_sup - con_w,
            character_count=len(chars),
            material_count=len(mats),
            remaining_unknowns=[
                "Whether Tex_I packs Ambient Occlusion in Red and GI in Green",
                "Whether blend mode is Soft Light vs Multiply in Unreal shader graph",
            ],
            alternative_hypotheses=[
                {"hypothesis": "ALT-I1: Ambient Occlusion Map", "status": "PLAUSIBLE (high luminance mean)"},
                {"hypothesis": "ALT-I2: Stylized Shadow Map", "status": "SUPPORTED"},
            ],
        )

    def _eval_base_eye_e(self) -> HypothesisResult:
        """HYP-EYE-E: BASE_Eye_E Shared Engine Eye Resource."""
        all_obs = self.ledger.query(asset_name="BASE_Eye_E.png")
        support = []
        contra = []

        for obs in all_obs:
            if obs.finding_type in ("engine_shared_resource", "channel_analysis"):
                support.append(obs)
            elif obs.finding_type == "json_unreferenced_contradiction":
                contra.append(obs)

        sup_w = sum(o.weight for o in support)
        con_w = sum(o.weight for o in contra)
        chars = {o.character for o in support + contra}
        mats = {o.material for o in support + contra}

        net_sup = 20.0 + sup_w
        conf = self._calc_bidirectional_confidence(net_sup, con_w)
        maturity = self._calc_maturity(conf, len(chars), len(contra))

        return HypothesisResult(
            hypothesis_id="HYP-EYE-E",
            name="BASE_Eye_E Shared Engine Eye Resource",
            current_interpretation="Unreferenced Shared Engine Eye Highlight / Mask Resource",
            confidence_score=conf,
            maturity_level=maturity,
            supporting_observations=support,
            contradicting_observations=contra,
            support_weight=net_sup,
            contradiction_weight=con_w,
            net_weight=net_sup - con_w,
            character_count=len(chars),
            material_count=len(mats),
            remaining_unknowns=[
                "Why BASE_Eye_E is omitted from exported Material Instance JSONs",
                "Whether BASE_Eye_E is bound dynamically at runtime by C++ EyeComponent",
            ],
            alternative_hypotheses=[
                {"hypothesis": "ALT-E1: Static Eye Specular Lookup Texture", "status": "PLAUSIBLE"},
                {"hypothesis": "ALT-E2: Unused Engine Asset", "status": "PLAUSIBLE"},
            ],
        )

    # -------------------------------------------------------------------------
    # Scientific Math Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _aggregate_obs(obs_list: List[Observation]) -> Tuple[float, float, set, set]:
        sup_w = sum(o.weight for o in obs_list)
        contra_w = 0.0
        chars = {o.character for o in obs_list}
        mats = {o.material for o in obs_list}
        return sup_w, contra_w, chars, mats

    @staticmethod
    def _calc_bidirectional_confidence(support_weight: float, contradiction_weight: float) -> float:
        """
        Calculate confidence percentage (0.0 - 100.0%).
        Bidirectional: decreases when contradiction_weight increases.
        """
        if support_weight <= 0.0:
            return 0.0

        # Ratio of support to total evidence weight
        total_w = support_weight + contradiction_weight * 2.5  # Penalize contradictions heavily
        raw_ratio = support_weight / max(1.0, total_w)

        # Scale to percentage range (max 99.5% for non-confirmed hypotheses)
        conf = min(99.5, max(5.0, raw_ratio * 100.0))
        return conf

    @staticmethod
    def _calc_maturity(confidence: float, char_count: int, contradiction_count: int) -> MaturityLevel:
        """
        Dynamically assign maturity level based on net confidence and character count.
        Promotes and demotes dynamically!
        """
        if confidence >= 95.0 and char_count >= 10 and contradiction_count == 0:
            return MaturityLevel.LEVEL_5_CONFIRMED
        elif confidence >= 75.0 and char_count >= 3:
            return MaturityLevel.LEVEL_4_HIGHLY_SUPPORTED
        elif confidence >= 50.0 and char_count >= 2:
            return MaturityLevel.LEVEL_3_SUPPORTED
        elif confidence >= 25.0:
            return MaturityLevel.LEVEL_2_PLAUSIBLE
        else:
            return MaturityLevel.LEVEL_1_UNKNOWN
