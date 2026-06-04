"""Assemble measured fault-injection files into the ``run_kfold`` input contract.

Each measurement file describes ONE injected coupling::

    {"origin": "<svc>::<KNOB>", "runs": [[svc, ...], ...]}

where ``runs`` is the per-window set of services that breached the SLO during a
replicated journey probe (``groundtruth.journey_probe``). This script folds a set
of such files into the three artifacts ``scripts/run_kfold.py`` consumes:

  - ground-truth  : ``[{scenario_id: "transfer-<slug>", impacted, stratum: "hidden"}]``
    The ``impacted`` blast is the 2/3-majority aggregate over that coupling's runs
    (the denoised transfer label; same MAJORITY rule as ``aggregate_runs``).
  - incidents     : one incident PER measurement window
    (``{origin, impacted, scenario_id: "teach-<slug>"}``). One incident per window
    gives the harness a genuine within-coupling learning curve rather than a single
    pre-aggregated outcome.
  - couplings     : the explicit list of coupling origin config-nodes.

In every case the INJECTED service itself (the ``<svc>`` of the origin node) is
excluded from its own blast and from each incident's impacted set: a latency/CPU
knob on service S trivially degrades S, so counting S would inflate the score with
a self-impact the engine is never asked to predict.

Scenario ids come from ``owm.scenarios.coupling_scenarios`` (``transfer-<slug>`` /
``teach-<slug>``), so the assembled files line up byte-for-byte with the scenario
set the rest of the harness derives from the same coupling list.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from groundtruth.journey_probe import aggregate_runs
from owm.scenarios import coupling_scenarios

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = ROOT / "data" / "groundtruth"


def _origin_service(origin: str) -> str:
    """The service a coupling origin config-node belongs to (``"<svc>::<KNOB>"`` -> ``"<svc>"``)."""
    return origin.split("::")[0]


def _blast(meas: dict) -> list[str]:
    """The denoised binary blast for one measurement: the 2/3-majority aggregate over
    its runs, with the injected service excluded. Prefer recomputing from ``runs`` so
    the same MAJORITY rule always applies; fall back to a precomputed ``blast`` key."""
    osvc = _origin_service(meas["origin"])
    if meas.get("runs"):
        blast = aggregate_runs(meas["runs"])
    else:
        blast = set(meas.get("blast", []))
    return sorted(s for s in blast if s != osvc)


def assemble(meas_list: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Fold measurement dicts into ``(ground_truth, incidents, couplings)``.

    ``ground_truth`` has one transfer record per coupling; ``incidents`` has one
    teach record per measurement window; ``couplings`` is the ordered origin list.
    The injected service is excluded from every blast/impacted set.
    """
    couplings = [m["origin"] for m in meas_list]
    scn = coupling_scenarios(couplings)
    transfer_id = {s.touched_configs[0]: s.id for s in scn if s.role == "transfer"}
    teach_id = {s.touched_configs[0]: s.id for s in scn if s.role == "teach"}

    gt: list[dict] = []
    incidents: list[dict] = []
    for meas in meas_list:
        origin = meas["origin"]
        osvc = _origin_service(origin)
        gt.append({
            "scenario_id": transfer_id[origin],
            "impacted": _blast(meas),
            "stratum": "hidden",
        })
        for window in meas.get("runs", []):          # one incident per measurement window
            impacted = sorted(s for s in set(window) if s != osvc)
            incidents.append({
                "origin": origin,
                "impacted": impacted,
                "scenario_id": teach_id[origin],
            })
    return gt, incidents, couplings


def _load_meas(patterns: list[str]) -> list[dict]:
    """Expand file paths / globs (in the given order, de-duplicated) and load each as JSON."""
    paths: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        matches = sorted(glob.glob(pat)) or [pat]
        for p in matches:
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return [json.loads(Path(p).read_text()) for p in paths]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meas", nargs="+", required=True,
                    help="one or more measurement JSON files or globs ({origin, runs})")
    ap.add_argument("--out-gt", default=str(DEFAULT_GT_DIR / "ground_truth_kfold_live.json"),
                    help="output ground-truth JSON (transfer blasts)")
    ap.add_argument("--out-incidents", default=str(DEFAULT_GT_DIR / "incidents_kfold_live.json"),
                    help="output incident-stream JSON (one record per window)")
    ap.add_argument("--out-couplings", default=str(DEFAULT_GT_DIR / "couplings_kfold_live.json"),
                    help="output coupling origin list JSON")
    args = ap.parse_args()

    meas_list = _load_meas(args.meas)
    if not meas_list:
        raise SystemExit("no measurement files found")

    gt, incidents, couplings = assemble(meas_list)

    for path, payload in (
        (args.out_gt, gt),
        (args.out_incidents, incidents),
        (args.out_couplings, couplings),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))

    print(f"couplings: {couplings}")
    print("\nground truth (transfer blasts):")
    for g in gt:
        print(f"  {g['scenario_id']:48s} -> {g['impacted']}")
    print(f"\nincidents (per-window): {len(incidents)} across {len(couplings)} couplings")
    print(f"wrote:\n  {args.out_gt}\n  {args.out_incidents}\n  {args.out_couplings}")


if __name__ == "__main__":
    main()
