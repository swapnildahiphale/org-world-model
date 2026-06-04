"""Offline tests for the K-coupling harness.

No network/cluster: ``run_kfold(offline=True)`` swaps in a FakeClient. We use the
bundled ``ob_mini`` fixture as the tiny base (4 services, real parsed config
nodes + call edges), pick TWO genuinely-parsed config knobs as the couplings,
fabricate matching transfer ground truth and a multi-incident stream, and assert
the two-axis contract:

  PRIMARY  -- WITHIN-coupling deepening: folding a coupling's OWN incidents drives
              its transfer Brier DOWN (engine_post < engine_pre).
  SECONDARY -- LOCALITY: training on the OTHER coupling's incidents is a no-op
              (engine_cross_mean ~= ablation_mean), because couplings are
              independent COUPLES edges on otherwise-disconnected config nodes.
"""
import json

import pytest

from scripts.run_kfold import run_kfold

FIXTURE_REPO = "tests/fixtures/ob_mini"

# Two genuinely-parsed config nodes from ob_mini (see codegraph parse): both are
# unconnected knobs whose downstream blast the engine must LEARN.
C1 = "productcatalogservice::EXTRA_LATENCY"        # blast: frontend, recommendationservice
C2 = "currencyservice::DISABLE_PROFILER"           # blast: frontend
COUPLINGS = [C1, C2]

# Transfer-scenario ids the harness derives from each coupling (slug of the origin).
TRANSFER_C1 = "transfer-productcatalogservice-extra-latency"
TRANSFER_C2 = "transfer-currencyservice-disable-profiler"


def _ground_truth():
    return [
        {"scenario_id": TRANSFER_C1, "stratum": "hidden",
         "impacted": ["frontend", "recommendationservice"]},
        {"scenario_id": TRANSFER_C2, "stratum": "hidden",
         "impacted": ["frontend"]},
    ]


def _incidents():
    """Multi-pass stream: 2 incidents per coupling (4 total)."""
    return [
        {"origin": C1, "impacted": ["frontend", "recommendationservice"],
         "scenario_id": "teach-productcatalogservice-extra-latency"},
        {"origin": C1, "impacted": ["frontend", "recommendationservice"],
         "scenario_id": "teach-productcatalogservice-extra-latency"},
        {"origin": C2, "impacted": ["frontend"],
         "scenario_id": "teach-currencyservice-disable-profiler"},
        {"origin": C2, "impacted": ["frontend"],
         "scenario_id": "teach-currencyservice-disable-profiler"},
    ]


@pytest.fixture
def kfold_inputs(tmp_path):
    gt_path = tmp_path / "gt.json"
    inc_path = tmp_path / "incidents.json"
    gt_path.write_text(json.dumps(_ground_truth()))
    inc_path.write_text(json.dumps(_incidents()))
    return gt_path, inc_path, tmp_path / "out"


def _run(kfold_inputs, **overrides):
    gt_path, inc_path, out_dir = kfold_inputs
    kwargs = dict(
        repo_path=FIXTURE_REPO,
        ground_truth_path=gt_path,
        incidents_path=inc_path,
        couplings=COUPLINGS,
        offline=True,
        out_dir=out_dir,
    )
    kwargs.update(overrides)
    return run_kfold(**kwargs)


def test_run_kfold_returns_contracted_top_level_keys(kfold_inputs):
    res = _run(kfold_inputs)
    assert set(res.keys()) >= {
        "mode", "n_couplings", "per_fold", "aggregate", "gap_ci",
        "deepening", "locality_control",
    }
    assert res["mode"] == "offline-stub"
    assert res["n_couplings"] == len(COUPLINGS)


def test_per_fold_length_equals_k(kfold_inputs):
    res = _run(kfold_inputs)
    assert len(res["per_fold"]) == len(COUPLINGS)
    assert [pf["coupling"] for pf in res["per_fold"]] == COUPLINGS


def test_within_coupling_deepens_on_fixture(kfold_inputs):
    """PRIMARY headline: folding a coupling's OWN incidents lowers its transfer
    Brier. On this fixture both couplings' incidents impact their transfer
    services, so engine_post < engine_pre for every coupling."""
    res = _run(kfold_inputs)
    for pf in res["per_fold"]:
        assert pf["engine"] < pf["engine_pre"], pf["coupling"]
    # the deepening block reflects the same drop in aggregate
    dp = res["deepening"]
    assert dp["engine_post_mean"] < dp["engine_pre_mean"]


def test_learning_curve_length_is_within_coupling_incident_count_plus_one(kfold_inputs):
    res = _run(kfold_inputs)
    incidents = _incidents()
    for pf in res["per_fold"]:
        own = [i for i in incidents if i["origin"] == pf["coupling"]]
        assert len(pf["learning_curve"]) == len(own) + 1
        assert pf["learning_curve"][0] == pytest.approx(pf["engine_pre"])
        assert pf["learning_curve"][-1] == pytest.approx(pf["engine"])
    # multi-pass: curve is non-increasing (each own-incident only sharpens belief)
    for pf in res["per_fold"]:
        c = pf["learning_curve"]
        assert all(c[i + 1] <= c[i] + 1e-12 for i in range(len(c) - 1)), pf["coupling"]


def test_every_fold_reports_all_predictors(kfold_inputs):
    res = _run(kfold_inputs)
    for pf in res["per_fold"]:
        for key in ("engine", "engine_pre", "stateless_llm", "rag_llm", "bfs", "ablation"):
            assert key in pf
            assert isinstance(pf[key], float)


def test_ablation_does_not_deepen(kfold_inputs):
    """learning_enabled=False => a coupling's transfer Brier equals its pre value."""
    res = _run(kfold_inputs)
    for pf in res["per_fold"]:
        assert pf["ablation"] == pytest.approx(pf["engine_pre"])


def test_locality_control_is_flat(kfold_inputs):
    """SECONDARY control: training on the OTHER coupling's incidents neither helps
    nor hurts a held-out coupling — engine_cross_mean ~= ablation_mean."""
    res = _run(kfold_inputs)
    lc = res["locality_control"]
    assert lc["engine_cross_mean"] == pytest.approx(lc["ablation_mean"])
    # and it stays at the uninformed (pre) level, i.e. far above the deepened post
    assert lc["engine_cross_mean"] > res["deepening"]["engine_post_mean"]


def test_gap_ci_is_two_element_lo_hi(kfold_inputs):
    res = _run(kfold_inputs)
    ci = res["gap_ci"]
    assert isinstance(ci, list) and len(ci) == 2
    lo, hi = ci
    assert lo <= hi


def test_aggregate_has_a_mean_per_predictor(kfold_inputs):
    res = _run(kfold_inputs)
    assert set(res["aggregate"].keys()) == {
        "engine", "stateless_llm", "rag_llm", "bfs", "ablation"
    }
    for v in res["aggregate"].values():
        assert isinstance(v, float)


def test_artifacts_are_written_and_reloadable(kfold_inputs):
    res = _run(kfold_inputs)
    _, _, out_dir = kfold_inputs
    results_json = out_dir / "results_kfold.json"
    png = out_dir / "reliability_kfold.png"
    assert results_json.exists()
    assert png.exists() and png.stat().st_size > 0
    reloaded = json.loads(results_json.read_text())
    assert reloaded["n_couplings"] == res["n_couplings"]
    assert len(reloaded["per_fold"]) == len(COUPLINGS)
    assert "deepening" in reloaded and "locality_control" in reloaded


def test_couplings_default_to_distinct_incident_origins(kfold_inputs):
    """Omitting the couplings list derives the set from distinct incident origins."""
    res = _run(kfold_inputs, couplings=None)
    assert res["n_couplings"] == 2
    assert [pf["coupling"] for pf in res["per_fold"]] == [C1, C2]
