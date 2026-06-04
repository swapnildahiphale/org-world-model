from owm.groundtruth import Prediction, GroundTruth
from owm.eval import Bin, deepening, state_delta, plot_reliability


def test_deepening_brier_improves_after_learning():
    gt = [GroundTruth("transfer-1", {"frontend"}, "hidden")]
    pre = [Prediction("transfer-1", {"frontend": 0.0, "currencyservice": 0.0})]   # blind
    post = [Prediction("transfer-1", {"frontend": 0.9, "currencyservice": 0.0})]  # learned
    d = deepening(pre, post, gt)
    assert d["brier_post"] < d["brier_pre"]


def test_state_delta_positive_for_amplified_service():
    healthy = Prediction("s", {"frontend": 0.3})
    saturated = Prediction("s", {"frontend": 0.7})
    assert state_delta(healthy, saturated)["frontend"] > 0


def test_plot_reliability_writes_a_png(tmp_path):
    out = tmp_path / "reliability.png"
    plot_reliability([Bin(0.2, 0.0, 2), Bin(0.8, 1.0, 2)], out)
    assert out.exists() and out.stat().st_size > 0
