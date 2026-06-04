"""Schema the evaluator consumes: a model Prediction and an independent
GroundTruth record. RCAEval cases and live fault-injection runs (T13) are
loaded into the SAME GroundTruth shape, so eval is blind to the source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Prediction:
    scenario_id: str
    probs: dict[str, float] = field(default_factory=dict)  # service -> P(impact)

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probs.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class GroundTruth:
    scenario_id: str
    impacted: set[str]
    stratum: str  # "given" | "hidden"


def load_ground_truth(path) -> list[GroundTruth]:
    raw = json.loads(Path(path).read_text())
    return [
        GroundTruth(
            scenario_id=r["scenario_id"],
            impacted=set(r["impacted"]),
            stratum=r["stratum"],
        )
        for r in raw
    ]
