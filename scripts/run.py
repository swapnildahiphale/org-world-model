"""End-to-end runner. --offline uses the test fixtures with no network/LLM and
prints the first real number: the provenance-stratified Brier score.

The HIDDEN-stratum Brier is the headline (Rule 1). Pre-learning it is poor by
design — the engine has not yet observed the EXTRA_LATENCY coupling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from owm.codegraph import parse_repo
from owm.topology import seed_graph
from owm.propagate import predict_blast
from owm.groundtruth import Prediction, load_ground_truth
from owm.eval import brier_score
from owm.scenarios import Scenario, default_scenarios
from owm.context import context_pack
from owm.engine import OWMEngine
from owm.baselines import StatelessLLM, RAGLLM, BFSBaseline, FakeClient, SELF_CONSISTENCY_N
from owm.eval import brier_score, reliability_diagram, deepening, state_delta, plot_reliability
import copy

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _predict_all(cwm, scenarios) -> list[Prediction]:
    preds = []
    for sc in scenarios:
        probs = predict_blast(
            cwm,
            touched_services=sc.get("touched_services"),
            touched_configs=sc.get("touched_configs"),
            s_t=sc.get("s_t", "healthy"),
        )
        preds.append(Prediction(scenario_id=sc["scenario_id"], probs=probs))
    return preds


def run(*, repo_path=None, scenarios_path=None, ground_truth_path=None,
        offline=False, out_path=None) -> dict:
    if offline:
        repo_path = repo_path or (FIXTURES / "ob_mini")
        scenarios_path = scenarios_path or (FIXTURES / "scenarios_mini.json")
        ground_truth_path = ground_truth_path or (FIXTURES / "ground_truth_mini.json")

    cwm = seed_graph(parse_repo(repo_path))
    scenarios = json.loads(Path(scenarios_path).read_text())
    gt = load_ground_truth(ground_truth_path)
    preds = _predict_all(cwm, scenarios)

    result = {
        "n_scenarios": len(scenarios),
        "brier_overall": brier_score(preds, gt),
        "brier_given": brier_score(preds, gt, stratum="given"),
        "brier_hidden": brier_score(preds, gt, stratum="hidden"),
    }

    out_path = Path(out_path) if out_path else (ROOT / "results" / "results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


# A deterministic offline LLM: name the obviously-reachable services. It cannot
# know the hidden config coupling (no call edge) -> it misses the HIDDEN stratum,
# which is exactly the point. Used only with --offline.
_OFFLINE_LLM_REPLY = '["frontend","recommendationservice"]'


def _load_incidents(path):
    import json as _json
    return _json.loads(Path(path).read_text())


def run_full(*, offline=False, repo_path=None, ground_truth_path=None,
             incidents_path=None, scenarios=None, out_dir=None,
             llm_client=None, self_consistency_n: int = SELF_CONSISTENCY_N) -> dict:
    if offline:
        repo_path = repo_path or (FIXTURES / "ob_mini")
        ground_truth_path = ground_truth_path or (FIXTURES / "ground_truth_full.json")
        incidents_path = incidents_path or (FIXTURES / "incidents_mini.json")
        llm_client = llm_client or FakeClient(_OFFLINE_LLM_REPLY)
    scenarios = scenarios or default_scenarios()
    out_dir = Path(out_dir) if out_dir else (ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cwm = seed_graph(parse_repo(repo_path))
    gt = load_ground_truth(ground_truth_path)
    incidents = _load_incidents(incidents_path)

    scored = [s for s in scenarios if s.role != "teach"]   # never score TEACH (§8)
    teach = [s for s in scenarios if s.role == "teach"]

    # --- pre-learning predictions, all predictors ---
    engine = OWMEngine(copy.deepcopy(base_cwm), learning_enabled=True)
    ablation = OWMEngine(copy.deepcopy(base_cwm), learning_enabled=False)
    bfs = BFSBaseline(base_cwm)
    stateless = StatelessLLM(llm_client, n=self_consistency_n)
    rag = RAGLLM(llm_client, n=self_consistency_n)

    def predict_all(eng):
        out = {}
        for s in scored:
            pack = context_pack(eng.cwm, s, incidents=incidents)
            out.setdefault("engine", []).append(eng.predict(s, pack))
            out.setdefault("stateless_llm", []).append(stateless.predict(s, pack))
            out.setdefault("rag_llm", []).append(rag.predict(s, pack))
            out.setdefault("bfs", []).append(bfs.predict(s, pack))
        return out

    pre = predict_all(engine)
    pre_engine = list(pre["engine"])
    pre_ablation = [ablation.predict(s, context_pack(ablation.cwm, s, incidents)) for s in scored]

    # --- feed the TEACH incident(s) to engine + ablation ---
    by_id = {s.id: s for s in teach}
    for inc in incidents:
        sc = by_id.get(inc["scenario_id"])
        if sc is None:
            continue
        engine.learn(sc, set(inc["impacted"]))
        ablation.learn(sc, set(inc["impacted"]))   # no-op: learning disabled

    # --- post-learning predictions ---
    post_engine = [engine.predict(s, context_pack(engine.cwm, s, incidents)) for s in scored]
    post_ablation = [ablation.predict(s, context_pack(ablation.cwm, s, incidents)) for s in scored]

    hidden_brier = {
        "engine_pre": brier_score(pre_engine, gt, stratum="hidden"),
        "engine": brier_score(post_engine, gt, stratum="hidden"),
        "stateless_llm": brier_score(pre["stateless_llm"], gt, stratum="hidden"),
        "rag_llm": brier_score(pre["rag_llm"], gt, stratum="hidden"),
        "bfs": brier_score(pre["bfs"], gt, stratum="hidden"),
    }
    deep = {
        "engine": deepening(pre_engine, post_engine, gt),
        "engine_no_learning": deepening(pre_ablation, post_ablation, gt),
    }

    # --- state contrast on the transfer change ---
    transfer = next((s for s in scored if s.role == "transfer"), None)
    contrast = {}
    if transfer is not None:
        h = engine.predict(Scenario.from_dict({**transfer.to_dict(), "s_t": "healthy"}))
        sat = engine.predict(Scenario.from_dict({**transfer.to_dict(), "s_t": "saturated"}))
        contrast = state_delta(h, sat)

    given_brier = {
        "engine": brier_score(post_engine, gt, stratum="given"),
        "stateless_llm": brier_score(pre["stateless_llm"], gt, stratum="given"),
        "rag_llm": brier_score(pre["rag_llm"], gt, stratum="given"),
        "bfs": brier_score(pre["bfs"], gt, stratum="given"),
    }

    # --- artifacts ---
    plot_reliability(reliability_diagram(post_engine, gt, n_bins=5, stratum="hidden"),
                     out_dir / "reliability.png")
    result = {
        "mode": "offline-stub" if offline else "live",
        "note": ("OFFLINE: baselines use a canned stub LLM — NOT a substantive "
                 "comparison. Offline, trust only the deepening (engine_pre vs engine) "
                 "and the ablation. Run live (omit --offline, set an API key) for a "
                 "real engine-vs-LLM contest.") if offline else "live LLM baselines",
        "hidden_brier": hidden_brier,
        "given_brier": given_brier,
        "deepening": deep,
        "state_contrast": contrast,
        "n_scored": len(scored),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="run against bundled fixtures, no network/LLM")
    ap.add_argument("--full", action="store_true", help="run the full predictor matrix")
    ap.add_argument("--provider", choices=["anthropic", "claude-code"],
                    default=os.environ.get("OWM_LLM_PROVIDER", "claude-code"),
                    help="LLM provider for live runs (default: OWM_LLM_PROVIDER env or claude-code)")
    ap.add_argument("--self-consistency-n", type=int, dest="self_consistency_n",
                    default=SELF_CONSISTENCY_N,
                    help="number of self-consistency samples for LLM baselines")
    args = ap.parse_args()

    if args.full:
        # live runs (not --offline) need a real LLM client + an API key
        from owm.baselines import make_live_client
        client = None if args.offline else make_live_client(provider=args.provider)
        res = run_full(offline=args.offline, llm_client=client,
                       self_consistency_n=args.self_consistency_n)
        print(json.dumps(res, indent=2))
        print("\n>>> HIDDEN-stratum Brier by predictor:", res["hidden_brier"])
    else:
        res = run(offline=args.offline)
        print(json.dumps(res, indent=2))
        print("\n>>> HIDDEN-stratum Brier (the honest headline):", res["brier_hidden"])


if __name__ == "__main__":
    main()
