"""Map RCAEval RE1/RE2 Online Boutique fault cases into our GroundTruth schema.

NOTE: confirm the real RCAEval file schema against the download (spec §12). This
adapter expects records with case_id / impacted_services / optional stratum; if
the real layout differs, change the mapping here, never the GroundTruth shape.
"""
from __future__ import annotations

import json
from pathlib import Path

from owm.groundtruth import GroundTruth


def load_rcaeval(path) -> list[GroundTruth]:
    raw = json.loads(Path(path).read_text())
    return [
        GroundTruth(
            scenario_id=r["case_id"],
            impacted=set(r["impacted_services"]),
            stratum=r.get("stratum", "hidden"),
        )
        for r in raw
    ]
