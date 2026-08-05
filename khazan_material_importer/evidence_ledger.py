"""
evidence_ledger.py
==================
Scientific Evidence Ledger for Khazan Material Importer.

Design Philosophy
-----------------
* SEPARATES raw observations from interpretations.
* The Evidence Ledger stores ONLY objective, empirical, reproducible data.
  (e.g., "CT_NPC_Daprona_UpperA_R mean pixel value = 0.04, max = 0.88 across 14 materials").
* It NEVER stores conclusions or hypothesis names directly.
* Stored persistently as JSON ('evidence_ledger.json').
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    """A single empirical, objective data point recorded from assets."""
    obs_id: str                          # e.g., "OBS-0001"
    category: str                        # e.g., "Pixel Statistics", "Cross Character", "Parameter Usage"
    asset_name: str                      # e.g., "CT_NPC_Daprona_UpperA_R.png" or "C_NPC_Daprona_Eye.json"
    character: str                      # e.g., "Daphrona"
    material: str                       # e.g., "CM_NPC_Daprona_UpperA"
    finding_type: str                   # e.g., "extrema", "channel_correlation", "parameter_cooccurrence"
    raw_metrics: Dict[str, Any]         # e.g., {"mean": 0.04, "min": 0, "max": 32, "std": 0.012}
    weight: float = 1.0                 # Quality weight based on asset count / reproducibility
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Observation:
        return cls(**data)


class EvidenceLedger:
    """Persistent storage database for raw empirical observations."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.observations: List[Observation] = []
        self._counter = 1

    def add_observation(
        self,
        category: str,
        asset_name: str,
        character: str,
        material: str,
        finding_type: str,
        raw_metrics: Dict[str, Any],
        weight: float = 1.0,
    ) -> Observation:
        """Record a new raw empirical observation."""
        obs_id = f"OBS-{self._counter:04d}"
        self._counter += 1

        obs = Observation(
            obs_id=obs_id,
            category=category,
            asset_name=asset_name,
            character=character,
            material=material,
            finding_type=finding_type,
            raw_metrics=raw_metrics,
            weight=max(0.1, weight),
        )
        self.observations.append(obs)
        return obs

    def clear(self) -> None:
        """Reset observations."""
        self.observations.clear()
        self._counter = 1

    def query(
        self,
        category: Optional[str] = None,
        asset_name: Optional[str] = None,
        character: Optional[str] = None,
        material: Optional[str] = None,
        finding_type: Optional[str] = None,
    ) -> List[Observation]:
        """Filter observations matching all criteria."""
        results = []
        for obs in self.observations:
            if category and obs.category != category:
                continue
            if asset_name and obs.asset_name.lower() != asset_name.lower():
                continue
            if character and obs.character.lower() != character.lower():
                continue
            if material and obs.material.lower() != material.lower():
                continue
            if finding_type and obs.finding_type != finding_type:
                continue
            results.append(obs)
        return results

    def save(self, filepath: Optional[str] = None) -> str:
        """Save evidence ledger to disk as JSON."""
        target = filepath or self.storage_path
        if not target:
            raise ValueError("No storage filepath specified for EvidenceLedger.")

        os.makedirs(os.path.dirname(target), exist_ok=True)
        data = {
            "version": "1.0",
            "observation_count": len(self.observations),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "observations": [obs.to_dict() for obs in self.observations],
        }

        with open(target, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return target

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load evidence ledger from JSON on disk."""
        target = filepath or self.storage_path
        if not target or not os.path.exists(target):
            return False

        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.observations = [
            Observation.from_dict(item) for item in data.get("observations", [])
        ]
        self._counter = len(self.observations) + 1
        return True
