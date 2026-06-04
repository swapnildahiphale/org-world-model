"""Offline tests for the live-measurement assembler.

No cluster/network: we feed ``assemble`` a synthetic measurement dict and assert
the ``run_kfold`` input contract — origin-service exclusion, one incident per
window, the GT/incident record shapes, and ids that line up with
``owm.scenarios.coupling_scenarios``.
"""
import json

import pytest

from owm.scenarios import coupling_scenarios
from scripts.assemble_live import assemble, main

ORIGIN = "currencyservice::EXTRA_LATENCY"
OSVC = "currencyservice"

# 3 windows. currencyservice (the injected service) appears in 2/3 — it would clear
# the 2/3-majority blast threshold, so excluding it is load-bearing, not incidental.
MEAS = {
    "origin": ORIGIN,
    "runs": [
        ["currencyservice", "frontend"],
        ["frontend"],
        ["currencyservice", "frontend"],
    ],
}


def _assemble_one():
    gt, incidents, couplings = assemble([MEAS])
    return gt, incidents, couplings


def test_couplings_is_the_origin_list():
    _, _, couplings = _assemble_one()
    assert couplings == [ORIGIN]


def test_origin_service_excluded_from_gt_blast():
    gt, _, _ = _assemble_one()
    assert len(gt) == 1
    assert OSVC not in gt[0]["impacted"]
    # frontend degraded in every window and survives; currencyservice is dropped.
    assert gt[0]["impacted"] == ["frontend"]


def test_gt_record_shape():
    gt, _, _ = _assemble_one()
    rec = gt[0]
    assert set(rec.keys()) == {"scenario_id", "impacted", "stratum"}
    assert rec["scenario_id"].startswith("transfer-")
    assert rec["stratum"] == "hidden"
    assert isinstance(rec["impacted"], list)


def test_origin_service_excluded_from_every_incident():
    _, incidents, _ = _assemble_one()
    for inc in incidents:
        assert OSVC not in inc["impacted"]


def test_one_incident_per_window():
    _, incidents, _ = _assemble_one()
    assert len(incidents) == len(MEAS["runs"])
    # window-by-window, origin excluded: [frontend], [frontend], [frontend]
    assert [inc["impacted"] for inc in incidents] == [["frontend"], ["frontend"], ["frontend"]]


def test_incident_record_shape():
    _, incidents, _ = _assemble_one()
    for inc in incidents:
        assert set(inc.keys()) == {"origin", "impacted", "scenario_id"}
        assert inc["origin"] == ORIGIN
        assert inc["scenario_id"].startswith("teach-")


def test_ids_match_coupling_scenarios():
    gt, incidents, _ = _assemble_one()
    scn = coupling_scenarios([ORIGIN])
    transfer_id = next(s.id for s in scn if s.role == "transfer")
    teach_id = next(s.id for s in scn if s.role == "teach")
    assert gt[0]["scenario_id"] == transfer_id
    assert all(inc["scenario_id"] == teach_id for inc in incidents)


def test_main_writes_three_artifacts(tmp_path, monkeypatch):
    meas_path = tmp_path / "meas_c1.json"
    meas_path.write_text(json.dumps(MEAS))
    out_gt = tmp_path / "gt.json"
    out_inc = tmp_path / "incidents.json"
    out_coup = tmp_path / "couplings.json"
    monkeypatch.setattr("sys.argv", [
        "assemble_live", "--meas", str(meas_path),
        "--out-gt", str(out_gt), "--out-incidents", str(out_inc),
        "--out-couplings", str(out_coup),
    ])
    main()

    gt, incidents, couplings = assemble([MEAS])
    assert json.loads(out_gt.read_text()) == gt
    assert json.loads(out_inc.read_text()) == incidents
    assert json.loads(out_coup.read_text()) == couplings


def test_glob_loads_multiple_measurement_files(tmp_path, monkeypatch):
    """A glob expands to several measurement files -> couplings reflect each origin."""
    other = {"origin": "productcatalogservice::EXTRA_LATENCY",
             "runs": [["frontend", "recommendationservice"]]}
    (tmp_path / "meas_c1.json").write_text(json.dumps(MEAS))
    (tmp_path / "meas_c2.json").write_text(json.dumps(other))
    out_gt = tmp_path / "gt.json"
    out_inc = tmp_path / "incidents.json"
    out_coup = tmp_path / "couplings.json"
    monkeypatch.setattr("sys.argv", [
        "assemble_live", "--meas", str(tmp_path / "meas_c*.json"),
        "--out-gt", str(out_gt), "--out-incidents", str(out_inc),
        "--out-couplings", str(out_coup),
    ])
    main()
    assert json.loads(out_coup.read_text()) == [ORIGIN, other["origin"]]
