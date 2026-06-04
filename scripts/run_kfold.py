"""K-coupling, multi-pass evaluation of the OWM engine.

Where ``scripts/run.py run_full`` runs a single teach->transfer deepening, this
harness generalizes it to K hidden couplings along the axis the model actually
exercises.

MODEL FACT this is built around: each coupling is an INDEPENDENT learned COUPLES
edge, and ``seed_graph`` adds config nodes with NO edges. So an unlearned config
node is fully disconnected -> its transfer scenario predicts ~0 regardless of
what the engine has learned about OTHER couplings. That makes two distinct axes:

  PRIMARY (headline) -- WITHIN-coupling deepening. For each coupling f we fold
  ONLY f's own incidents (multi-pass) and watch f's transfer Brier drop. This is
  the real "deepen" claim: an unconnected knob, after observing its own
  incidents, becomes well-calibrated about its blast radius. engine_post is
  scored vs the LLM/BFS baselines and a learning-disabled ablation.

  SECONDARY (control) -- LOCALITY. For each f we instead train on every OTHER
  coupling's incidents and test f. Because the couplings are independent, this
  should be a no-op: engine_cross_mean ~= ablation_mean ~= engine_pre. Learning
  coupling A neither helps nor hurts coupling B. That is locality, not deepening.

Aggregates: mean Brier per predictor; a seeded bootstrap 95% CI of the per-
coupling (engine_post - stateless) gap; a pooled hidden-stratum reliability
diagram over the engine_post transfer predictions; and a top-level deepening
block (mean engine_pre vs mean engine_post across couplings).

The LLM/BFS baselines do not learn, so each transfer scenario's baseline
prediction is computed ONCE and reused (only the engine's state changes per
fold). Run live (real LLM) by default; ``--offline`` swaps in a FakeClient with
canned JSON so the whole thing runs with no network/cluster (used by the tests).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
from pathlib import Path

import yaml

from owm.codegraph import parse_repo
from owm.topology import seed_graph
from owm.groundtruth import GroundTruth, Prediction, load_ground_truth
from owm.scenarios import Scenario, coupling_scenarios, _slug
from owm.context import context_pack
from owm.engine import OWMEngine
from owm.baselines import (
    StatelessLLM, RAGLLM, BFSBaseline, FakeClient, make_live_client,
    SELF_CONSISTENCY_N,
)
from owm.eval import brier_score, reliability_diagram, plot_reliability

ROOT = Path(__file__).resolve().parents[1]

# Canned offline baseline reply: name the statically-reachable services. It cannot
# know any hidden config coupling (there is no call edge to it) -> it misses the
# HIDDEN stratum, exactly as the live stateless baseline is expected to. Used only
# with --offline; NOT a substantive engine-vs-LLM comparison.
_OFFLINE_LLM_REPLY = '["frontend","recommendationservice"]'

BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 1729


def _stage_repo(path: Path) -> Path:
    """Normalize the repo input to a dir ``parse_repo`` accepts.

    A directory is used as-is. A path to a ``kubernetes-manifests.yaml`` file is
    staged into a temp dir (mirrors ``scripts/live_run.py``) so the static parse
    reflects the actually-supplied topology.
    """
    path = Path(path)
    if path.is_dir():
        return path
    if path.is_file():
        d = Path(tempfile.mkdtemp(prefix="owm_kfold_repo_"))
        (d / "kubernetes-manifests.yaml").write_text(path.read_text())
        return d
    raise FileNotFoundError(f"repo manifest not found: {path}")


def _gt_for(gt: list[GroundTruth], scenario_id: str) -> list[GroundTruth]:
    return [g for g in gt if g.scenario_id == scenario_id]


def _teach_scenario_for(origin: str) -> Scenario:
    """A teach scenario the engine can learn() from for this incident origin."""
    return Scenario(id=f"teach-{_slug(origin)}", stratum="hidden", role="teach",
                    touched_configs=(origin,), s_t="healthy")


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]) over a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _bootstrap_ci(diffs: list[float], *, b: int = BOOTSTRAP_B,
                  seed: int = BOOTSTRAP_SEED) -> list[float]:
    """Seeded percentile bootstrap 95% CI of the mean of per-fold diffs."""
    if not diffs:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(b):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def run_kfold(*, repo_path, ground_truth_path, incidents_path,
              couplings=None, provider: str = "claude-code",
              self_consistency_n: int = SELF_CONSISTENCY_N,
              offline: bool = False, out_dir=None,
              llm_client=None) -> dict:
    """Run the K-coupling evaluation (within-coupling deepening = headline,
    hold-one-out = locality control); return the results dict.

    main() is a thin CLI wrapper around this so tests can call it directly.
    """
    out_dir = Path(out_dir) if out_dir else (ROOT / "results_kfold")
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = _stage_repo(Path(repo_path))
    base_cwm = seed_graph(parse_repo(repo))
    gt = load_ground_truth(ground_truth_path)
    incidents = json.loads(Path(incidents_path).read_text())

    # K couplings = explicit list, else the distinct incident origins (stable order).
    if couplings is None:
        couplings = []
        seen = set()
        for inc in incidents:
            o = inc["origin"]
            if o not in seen:
                seen.add(o)
                couplings.append(o)
    k = len(couplings)

    # The scored scenario per coupling is that coupling's TRANSFER change.
    transfer_by_coupling = {
        c: s for c, s in zip(
            couplings,
            (sc for sc in coupling_scenarios(couplings) if sc.role == "transfer"),
        )
    }

    if offline:
        llm_client = llm_client or FakeClient(_OFFLINE_LLM_REPLY)
    elif llm_client is None:
        llm_client = make_live_client(provider=provider)

    stateless = StatelessLLM(llm_client, n=self_consistency_n)
    rag = RAGLLM(llm_client, n=self_consistency_n)
    bfs = BFSBaseline(base_cwm)

    def _brier(pred, sc_gt) -> float:
        return brier_score([pred], sc_gt, stratum="hidden")

    def _engine_brier(eng, sc, sc_gt) -> float:
        return _brier(eng.predict(sc, context_pack(eng.cwm, sc, incidents=incidents)), sc_gt)

    # --- baselines: score each transfer ONCE and cache (they don't learn) ---
    baseline_brier: dict[str, dict[str, float]] = {}
    for c in couplings:
        sc = transfer_by_coupling[c]
        sc_gt = _gt_for(gt, sc.id)
        # baselines see the SAME pack the engine would (symmetry); incidents are the
        # full stream — the stateless baseline ignores them, RAG reads them.
        pack = context_pack(base_cwm, sc, incidents=incidents)
        baseline_brier[c] = {
            "stateless_llm": _brier(stateless.predict(sc, pack), sc_gt),
            "rag_llm": _brier(rag.predict(sc, pack), sc_gt),
            "bfs": _brier(bfs.predict(sc, pack), sc_gt),
        }

    def _incidents_for(origin: str) -> list[dict]:
        return [inc for inc in incidents if inc["origin"] == origin]

    # =========================================================================
    # PRIMARY (headline): WITHIN-coupling deepening — fold f's OWN incidents.
    # =========================================================================
    per_fold: list[dict] = []
    gap_diffs: list[float] = []          # per-coupling (engine_post - stateless)
    pre_vals: list[float] = []
    post_vals: list[float] = []
    agg: dict[str, list[float]] = {
        "engine": [], "stateless_llm": [], "rag_llm": [], "bfs": [], "ablation": [],
    }
    pooled_preds: list[Prediction] = []  # engine_post transfer predictions (calibration)
    pooled_gt: list[GroundTruth] = []

    for coupling in couplings:
        sc = transfer_by_coupling[coupling]
        sc_gt = _gt_for(gt, sc.id)

        engine = OWMEngine(copy.deepcopy(base_cwm), learning_enabled=True)
        ablation = OWMEngine(copy.deepcopy(base_cwm), learning_enabled=False)

        # pre-learning transfer Brier (curve element 0)
        engine_pre = _engine_brier(engine, sc, sc_gt)
        learning_curve = [engine_pre]

        # multi-pass: fold ONLY this coupling's own incidents, in order.
        for inc in _incidents_for(coupling):
            teach = _teach_scenario_for(inc["origin"])
            engine.learn(teach, set(inc["impacted"]))
            ablation.learn(teach, set(inc["impacted"]))     # no-op: learning disabled
            learning_curve.append(_engine_brier(engine, sc, sc_gt))

        engine_post_pred = engine.predict(sc, context_pack(engine.cwm, sc, incidents=incidents))
        engine_post = _brier(engine_post_pred, sc_gt)
        ablation_brier = _engine_brier(ablation, sc, sc_gt)

        cached = baseline_brier[coupling]
        per_fold.append({
            "coupling": coupling,
            "engine_pre": engine_pre,
            "engine": engine_post,
            "stateless_llm": cached["stateless_llm"],
            "rag_llm": cached["rag_llm"],
            "bfs": cached["bfs"],
            "ablation": ablation_brier,
            "learning_curve": learning_curve,
        })

        agg["engine"].append(engine_post)
        agg["stateless_llm"].append(cached["stateless_llm"])
        agg["rag_llm"].append(cached["rag_llm"])
        agg["bfs"].append(cached["bfs"])
        agg["ablation"].append(ablation_brier)
        gap_diffs.append(engine_post - cached["stateless_llm"])
        pre_vals.append(engine_pre)
        post_vals.append(engine_post)

        pooled_preds.append(engine_post_pred)
        pooled_gt.extend(sc_gt)

    # =========================================================================
    # SECONDARY (control): LOCALITY — train on OTHER couplings, test f.
    # Independent couplings => this should match the ablation/pre value.
    # =========================================================================
    cross_vals: list[float] = []
    locality_ablation_vals: list[float] = []
    for coupling in couplings:
        sc = transfer_by_coupling[coupling]
        sc_gt = _gt_for(gt, sc.id)

        engine = OWMEngine(copy.deepcopy(base_cwm), learning_enabled=True)
        ablation = OWMEngine(copy.deepcopy(base_cwm), learning_enabled=False)
        for inc in incidents:
            if inc["origin"] == coupling:
                continue
            teach = _teach_scenario_for(inc["origin"])
            engine.learn(teach, set(inc["impacted"]))
            ablation.learn(teach, set(inc["impacted"]))     # no-op
        cross_vals.append(_engine_brier(engine, sc, sc_gt))
        locality_ablation_vals.append(_engine_brier(ablation, sc, sc_gt))

    def _mean(xs: list[float]) -> float:
        return (sum(xs) / len(xs)) if xs else float("nan")

    aggregate = {name: _mean(vals) for name, vals in agg.items()}
    gap_ci = _bootstrap_ci(gap_diffs)

    # pooled calibration over the engine_post transfer predictions (HIDDEN)
    bins = reliability_diagram(pooled_preds, pooled_gt, n_bins=5, stratum="hidden")
    plot_reliability(bins, out_dir / "reliability_kfold.png")

    result = {
        "mode": "offline-stub" if offline else "live",
        "n_couplings": k,
        "per_fold": per_fold,
        "aggregate": aggregate,
        "gap_ci": gap_ci,
        "deepening": {
            "engine_pre_mean": _mean(pre_vals),
            "engine_post_mean": _mean(post_vals),
        },
        "locality_control": {
            "engine_cross_mean": _mean(cross_vals),
            "ablation_mean": _mean(locality_ablation_vals),
        },
    }
    (out_dir / "results_kfold.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-manifest", required=True, dest="repo_manifest",
                    help="dir with kubernetes-manifests.yaml, or path to that manifest")
    ap.add_argument("--ground-truth", required=True, dest="ground_truth",
                    help="ground-truth JSON (transfer scenarios + strata)")
    ap.add_argument("--incidents", required=True,
                    help="incident-stream JSON ({origin, impacted, scenario_id} records)")
    ap.add_argument("--couplings", default=None,
                    help="optional JSON file: list of coupling origin strings "
                         "(default: distinct incident origins)")
    ap.add_argument("--provider", choices=["anthropic", "claude-code"],
                    default=os.environ.get("OWM_LLM_PROVIDER", "claude-code"),
                    help="LLM provider for live runs (default: env OWM_LLM_PROVIDER or claude-code)")
    ap.add_argument("--self-consistency-n", type=int, dest="self_consistency_n",
                    default=SELF_CONSISTENCY_N,
                    help="self-consistency samples for the LLM baselines")
    ap.add_argument("--offline", action="store_true",
                    help="use a canned FakeClient (no network/cluster)")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    couplings = None
    if args.couplings:
        couplings = json.loads(Path(args.couplings).read_text())

    res = run_kfold(
        repo_path=args.repo_manifest,
        ground_truth_path=args.ground_truth,
        incidents_path=args.incidents,
        couplings=couplings,
        provider=args.provider,
        self_consistency_n=args.self_consistency_n,
        offline=args.offline,
        out_dir=args.out,
    )
    print(json.dumps(res, indent=2))
    dp = res["deepening"]
    lc = res["locality_control"]
    print(f"\n>>> {res['n_couplings']} couplings | within-coupling deepening "
          f"(headline): mean Brier {dp['engine_pre_mean']:.4f} -> {dp['engine_post_mean']:.4f}")
    print(f">>> aggregate Brier by predictor: {res['aggregate']}")
    print(f">>> engine_post-vs-stateless gap 95% CI: {res['gap_ci']}")
    print(f">>> locality control: engine_cross {lc['engine_cross_mean']:.4f} "
          f"~= ablation {lc['ablation_mean']:.4f} (learning OTHER couplings is a no-op)")


if __name__ == "__main__":
    main()
