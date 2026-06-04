"""Aggregate replicated journey-probe runs into one GroundTruth record.

Each run is the set of services that breached the pre-registered SLO during a
browse/cart/checkout probe. A service is 'in blast radius' iff it degraded in at
least two-thirds of runs (the label-noise control from spec §7).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from owm.groundtruth import GroundTruth

MAJORITY = 2.0 / 3.0


def aggregate_runs(runs, threshold: float = MAJORITY) -> set[str]:
    counts: Counter = Counter()
    for run in runs:
        for svc in set(run):
            counts[svc] += 1
    n = len(runs)
    if n == 0:
        return set()
    return {svc for svc, k in counts.items() if k / n >= threshold}


def probe_file_to_ground_truth(path) -> GroundTruth:
    rec = json.loads(Path(path).read_text())
    return GroundTruth(
        scenario_id=rec["scenario_id"],
        impacted=aggregate_runs(rec["runs"]),
        stratum=rec.get("stratum", "hidden"),
    )
