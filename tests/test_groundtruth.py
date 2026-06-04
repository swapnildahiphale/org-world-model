from pathlib import Path
from owm.groundtruth import GroundTruth, Prediction, load_ground_truth

FIXTURE = Path(__file__).parent / "fixtures" / "ground_truth_mini.json"


def test_load_ground_truth():
    gt = load_ground_truth(FIXTURE)
    by_id = {g.scenario_id: g for g in gt}
    assert by_id["hidden-extra-latency"].stratum == "hidden"
    assert by_id["hidden-extra-latency"].impacted == {"frontend", "recommendationservice"}
    assert by_id["given-pc-handler"].stratum == "given"


def test_prediction_holds_per_service_probabilities():
    p = Prediction(scenario_id="x", probs={"frontend": 0.6, "currencyservice": 0.0})
    assert p.probs["frontend"] == 0.6
    # ranked() returns services sorted by descending probability
    assert p.ranked()[0][0] == "frontend"
