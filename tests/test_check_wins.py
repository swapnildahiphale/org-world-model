from pathlib import Path
from scripts.assemble_ground_truth import assemble
from scripts.check_wins import evaluate_wins

FIX = Path(__file__).parent / "fixtures"


def test_assemble_merges_rcaeval_and_probe_records():
    gt = assemble(rcaeval_path=FIX / "rcaeval_sample.json",
                  probe_paths=[FIX / "probe_runs_sample.json"])
    ids = {g.scenario_id for g in gt}
    assert "OB-productcatalog-latency" in ids        # from RCAEval
    assert "live-extra-latency" in ids               # from a live probe


def test_evaluate_wins_reads_a_results_dict():
    results = {
        "hidden_brier": {"engine": 0.05, "stateless_llm": 0.4, "rag_llm": 0.3, "bfs": 0.5},
        "deepening": {
            "engine": {"brier_pre": 0.5, "brier_post": 0.05},
            "engine_no_learning": {"brier_pre": 0.5, "brier_post": 0.5},
        },
        "state_contrast": {"frontend": 0.4},
    }
    wins = evaluate_wins(results)
    assert wins["engine_beats_baselines"] is True
    assert wins["engine_deepens"] is True
    assert wins["ablation_flat"] is True
    assert wins["state_amplifies"] is True


def test_evaluate_wins_is_honest_about_losses():
    results = {
        "hidden_brier": {"engine": 0.45, "stateless_llm": 0.4, "rag_llm": 0.3, "bfs": 0.5},
        "deepening": {"engine": {"brier_pre": 0.4, "brier_post": 0.45},
                      "engine_no_learning": {"brier_pre": 0.5, "brier_post": 0.5}},
        "state_contrast": {},
    }
    wins = evaluate_wins(results)
    assert wins["engine_beats_baselines"] is False
    assert wins["engine_deepens"] is False
    assert wins["state_amplifies"] is False
