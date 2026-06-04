from pathlib import Path
from owm.groundtruth import GroundTruth
from groundtruth.rcaeval_loader import load_rcaeval
from groundtruth.journey_probe import aggregate_runs, probe_file_to_ground_truth

FIX = Path(__file__).parent / "fixtures"


def test_rcaeval_maps_into_ground_truth_schema():
    gt = load_rcaeval(FIX / "rcaeval_sample.json")
    assert all(isinstance(g, GroundTruth) for g in gt)
    by_id = {g.scenario_id: g for g in gt}
    assert by_id["OB-productcatalog-latency"].impacted == {"frontend", "recommendationservice"}
    assert by_id["OB-cartservice-cpu"].stratum == "hidden"


def test_aggregate_runs_uses_two_thirds_majority():
    # frontend 3/3, recommendation 2/3 (kept), currency 1/3 (dropped)
    runs = [["frontend", "recommendationservice"],
            ["frontend", "recommendationservice"],
            ["frontend", "currencyservice"]]
    assert aggregate_runs(runs) == {"frontend", "recommendationservice"}


def test_probe_file_becomes_one_ground_truth_record():
    g = probe_file_to_ground_truth(FIX / "probe_runs_sample.json")
    assert g.scenario_id == "live-extra-latency"
    assert g.impacted == {"frontend", "recommendationservice"}
    assert g.stratum == "hidden"
