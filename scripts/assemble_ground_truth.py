"""Merge RCAEval cases + live journey-probe records into one ground-truth set."""
from __future__ import annotations

import json
from pathlib import Path

from owm.groundtruth import GroundTruth
from groundtruth.rcaeval_loader import load_rcaeval
from groundtruth.journey_probe import probe_file_to_ground_truth


def assemble(rcaeval_path=None, probe_paths=()) -> list[GroundTruth]:
    gt: list[GroundTruth] = []
    if rcaeval_path:
        gt += load_rcaeval(rcaeval_path)
    for p in probe_paths:
        gt.append(probe_file_to_ground_truth(p))
    return gt


def write_ground_truth(gt: list[GroundTruth], path) -> None:
    rows = [{"scenario_id": g.scenario_id, "impacted": sorted(g.impacted),
             "stratum": g.stratum} for g in gt]
    Path(path).write_text(json.dumps(rows, indent=2))
