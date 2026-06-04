"""Offline tests for the multi-pass convergence script.

No cluster/network and NO dependence on the Online Boutique manifest: we build a
tiny synthetic base CausalWorldModel with ``seed_graph`` over a minimal
``ParsedCode`` (two services + the injected coupling's config node, deliberately
NO call edges so the config node is disconnected and its learned COUPLES weight
surfaces directly as the predicted blast probability) and inject it into ``run``.

The deterministic measurement degrades ``frontend`` in every window, so we can
assert the headline behavior exactly: frontend's learned weight climbs toward
~1.0, the engine's per-window Brier drops pass-over-pass while the ablation's
stays flat, and the plot + JSON are written.
"""
import json

import matplotlib
import pytest

from owm.codegraph import ParsedCode
from owm.graph import Kind, Rel
from owm.topology import seed_graph
from scripts.multipass import run

matplotlib.use("Agg")

ORIGIN = "currencyservice::EXTRA_LATENCY"

# 4 windows, frontend degrades in every one (currencyservice = the injected service,
# excluded by run()); recommendationservice stays clean throughout.
MEAS = {
    "origin": ORIGIN,
    "runs": [
        ["currencyservice", "frontend"],
        ["currencyservice", "frontend"],
        ["currencyservice", "frontend"],
        ["currencyservice", "frontend"],
    ],
}


@pytest.fixture
def base_cwm():
    """Tiny seeded graph: two services + the origin config node, no call edges.

    With no CALLS path into frontend, its predicted blast probability equals the
    learned ``origin --couples--> frontend`` weight, which makes the convergence
    assertions exact and independent of any real manifest topology."""
    pc = ParsedCode(
        services=["frontend", "recommendationservice"],
        config_nodes=[ORIGIN],
    )
    cwm = seed_graph(pc)
    # sanity: the config origin is a node and starts fully disconnected (no coupling yet)
    assert ORIGIN in cwm.nodes_of(Kind.CONFIG)
    assert cwm.g.out_degree(ORIGIN) == 0
    return cwm


def _run(base_cwm, tmp_path):
    out_plot = tmp_path / "multipass.png"
    out_json = tmp_path / "multipass.json"
    result = run(MEAS, cwm=base_cwm, out_plot=out_plot, out_json=out_json)
    return result, out_plot, out_json


def test_frontend_weight_trajectory_climbs_toward_one(base_cwm, tmp_path):
    result, _, _ = _run(base_cwm, tmp_path)
    traj = result["weight_trajectories"]["frontend"]
    assert traj[0] == 0.0                              # no coupling before any pass
    assert all(traj[i + 1] >= traj[i] for i in range(len(traj) - 1))  # monotone up
    assert traj[-1] > 0.8                              # converging toward 1.0
    # frontend degraded in EVERY window -> empirical rate 1.0 (the calibration target)
    assert result["empirical_rate"]["frontend"] == pytest.approx(1.0)


def test_per_window_brier_decreases_pass_over_pass(base_cwm, tmp_path):
    result, _, _ = _run(base_cwm, tmp_path)
    pwb = result["engine_per_window_brier"]
    assert all(pwb[i + 1] <= pwb[i] + 1e-12 for i in range(len(pwb) - 1))  # non-increasing
    assert pwb[-1] < pwb[0]                            # net improvement from learning


def test_ablation_per_window_brier_is_flat(base_cwm, tmp_path):
    result, _, _ = _run(base_cwm, tmp_path)
    abl = result["ablation_per_window_brier"]
    assert all(v == pytest.approx(abl[0]) for v in abl)  # learning off => beliefs never move


def test_passes_count_is_windows_plus_one(base_cwm, tmp_path):
    result, _, _ = _run(base_cwm, tmp_path)
    assert result["passes"] == list(range(len(MEAS["runs"]) + 1))
    for series in ("engine_per_window_brier", "ablation_per_window_brier",
                   "engine_binary_brier"):
        assert len(result[series]) == len(MEAS["runs"]) + 1
    for traj in result["weight_trajectories"].values():
        assert len(traj) == len(MEAS["runs"]) + 1


def test_blast_excludes_injected_service(base_cwm, tmp_path):
    result, _, _ = _run(base_cwm, tmp_path)
    assert "currencyservice" not in result["blast"]
    assert result["blast"] == ["frontend"]             # 2/3-majority denoised, origin dropped


def test_artifacts_written(base_cwm, tmp_path):
    _, out_plot, out_json = _run(base_cwm, tmp_path)
    assert out_plot.exists() and out_plot.stat().st_size > 0
    reloaded = json.loads(out_json.read_text())
    assert reloaded["origin"] == ORIGIN
    assert "weight_trajectories" in reloaded


def test_injected_cwm_is_not_mutated(base_cwm, tmp_path):
    """run() deep-copies the injected graph, so the caller's cwm gains no learned edges."""
    before = base_cwm.g.number_of_edges()
    _run(base_cwm, tmp_path)
    assert base_cwm.g.number_of_edges() == before
    assert not base_cwm.g.has_edge(ORIGIN, "frontend", key=Rel.COUPLES)
