from owm.groundtruth import Prediction, GroundTruth
from owm.eval import brier_score, reliability_diagram, points


def _data():
    preds = [
        Prediction("s1", {"a": 1.0, "b": 0.0}),   # both correct
        Prediction("s2", {"a": 0.0, "b": 1.0}),   # both wrong
    ]
    gt = [
        GroundTruth("s1", {"a"}, "given"),
        GroundTruth("s2", {"a"}, "hidden"),   # b predicted 1.0 but only a impacted
    ]
    return preds, gt


def test_points_are_scenario_candidate_pairs_with_outcomes():
    preds, gt = _data()
    pts = points(preds, gt)            # list of (prob, outcome, stratum)
    assert (1.0, 1, "given") in pts    # s1/a predicted 1, impacted
    assert (0.0, 0, "given") in pts    # s1/b predicted 0, not impacted
    assert (0.0, 1, "hidden") in pts   # s2/a predicted 0, impacted (a miss)
    assert (1.0, 0, "hidden") in pts   # s2/b predicted 1, not impacted (a false alarm)


def test_brier_overall_and_stratified():
    preds, gt = _data()
    # overall: s1 contributes 0,0 ; s2 contributes (0-1)^2=1, (1-0)^2=1 -> mean of [0,0,1,1]=0.5
    assert abs(brier_score(preds, gt) - 0.5) < 1e-9
    # given stratum only: [0,0] -> 0.0
    assert abs(brier_score(preds, gt, stratum="given") - 0.0) < 1e-9
    # hidden stratum only: [1,1] -> 1.0  (the honest headline: engine is wrong pre-learning)
    assert abs(brier_score(preds, gt, stratum="hidden") - 1.0) < 1e-9


def test_reliability_equal_frequency_bins():
    # four points; 2 equal-frequency bins => 2 points each
    preds = [Prediction("s", {"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9})]
    gt = [GroundTruth("s", {"c", "d"}, "given")]   # high-prob ones are the impacted ones
    bins = reliability_diagram(preds, gt, n_bins=2)
    assert len(bins) == 2
    lo, hi = bins
    assert lo.count == 2 and hi.count == 2
    assert abs(lo.mean_pred - 0.15) < 1e-9 and lo.observed_freq == 0.0
    assert abs(hi.mean_pred - 0.85) < 1e-9 and hi.observed_freq == 1.0
