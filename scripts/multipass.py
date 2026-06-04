"""Deep multi-pass convergence on a single learned coupling.

Where ``scripts/run_kfold.py`` scores many couplings at one folding depth, this
script zooms in on ONE coupling and folds its measurement windows one pass at a
time, recording the engine's belief as it converges. A measurement file is::

    {"origin": "<svc>::<KNOB>", "runs": [[svc, ...], ...]}

Each ``runs`` window is the set of services that breached the SLO in one replicated
journey probe. We fold each window as ONE learning pass into an OWMEngine and track,
per pass:

  - the learned COUPLES edge weight (Beta posterior mean) for EVERY service that
    ever degraded. With incidents folded in, each weight converges to that service's
    empirical impact rate over the windows — calibration, not memorization (a service
    that degrades in 2/3 of probes should settle near 0.67, not 1.0).
  - the per-window Brier of the engine's transfer prediction (a proper probabilistic
    score: a calibrated 0.67 for an intermittent service beats a hard 0 or 1).

vs a learning-OFF ablation (``learning_enabled=False``), whose per-window Brier is
flat because its beliefs never move. The injected service is excluded throughout (a
knob trivially degrades its own service; that self-impact is not what we predict).

Emits a 2-panel matplotlib (Agg) figure — top: per-window Brier, engine vs flat
ablation; bottom: per-service learned-weight trajectories with dashed empirical-rate
reference lines — plus a JSON dump of every recorded series.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt

from groundtruth.journey_probe import aggregate_runs
from owm.codegraph import parse_repo
from owm.engine import OWMEngine
from owm.eval import brier_score
from owm.graph import Rel
from owm.groundtruth import GroundTruth
from owm.scenarios import Scenario, _slug
from owm.topology import seed_graph

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_MANIFEST = ROOT / "data" / "online_boutique" / "repo" / "release"

# Stable per-service colors for the weight-trajectory panel; any service outside
# this map falls back to slate.
_COLORS = {
    "frontend": "#0f766e",
    "recommendationservice": "#b45309",
    "checkoutservice": "#7c3aed",
}
_FALLBACK_COLOR = "#334155"


def _coupling_weight(engine: OWMEngine, origin: str, service: str) -> float:
    """Current learned COUPLES edge weight ``origin -> service`` (0.0 if no such edge)."""
    g = engine.cwm.g
    if g.has_edge(origin, service, key=Rel.COUPLES):
        return g.get_edge_data(origin, service, key=Rel.COUPLES)["weight"]
    return 0.0


def _per_window_brier(engine: OWMEngine, transfer: Scenario, windows: list[list[str]]) -> float:
    """Mean Brier of the engine's transfer prediction scored against each individual
    window (per (window, candidate-service)). Rewards calibrated probabilities on the
    services that degrade intermittently across windows."""
    probs = engine.predict(transfer).probs
    services = list(probs)
    if not services or not windows:
        return float("nan")
    total = 0.0
    n = 0
    for window in windows:
        present = set(window)
        for s in services:
            outcome = 1.0 if s in present else 0.0
            total += (probs.get(s, 0.0) - outcome) ** 2
            n += 1
    return total / n


def run(meas: dict, *, repo_manifest=DEFAULT_REPO_MANIFEST, cwm=None,
        out_plot=None, out_json=None) -> dict:
    """Fold ``meas``'s windows one pass at a time and record convergence.

    The base graph is either the injected ``cwm`` (a pre-built CausalWorldModel —
    used by tests so no Online Boutique manifest is needed) or, if ``cwm is None``,
    seeded from ``repo_manifest`` (a dir with ``kubernetes-manifests.yaml``). The
    injected ``cwm`` is deep-copied per engine, so the caller's graph is untouched.

    Returns the results dict; also writes the 2-panel plot to ``out_plot`` and the
    series JSON to ``out_json`` when those paths are given. ``main()`` is a thin CLI
    wrapper around this.
    """
    origin = meas["origin"]
    osvc = origin.split("::")[0]
    runs = meas["runs"]

    # per-pass incidents and the denoised binary blast, both with the injected
    # service excluded (a knob trivially degrades its own service).
    windows = [[s for s in run_ if s != osvc] for run_ in runs]
    blast = sorted(s for s in aggregate_runs(runs) if s != osvc)
    tracked = sorted({s for w in windows for s in w})            # every service that ever degraded
    rate = {s: sum(s in w for w in windows) / len(windows) for s in tracked} if windows else {}

    base = copy.deepcopy(cwm) if cwm is not None else seed_graph(parse_repo(Path(repo_manifest)))
    transfer = Scenario(id=f"transfer-{_slug(origin)}", stratum="hidden", role="transfer",
                        touched_configs=(origin,))
    teach = Scenario(id=f"teach-{_slug(origin)}", stratum="hidden", role="teach",
                     touched_configs=(origin,))
    gt_bin = [GroundTruth(transfer.id, set(blast), "hidden")]

    engine = OWMEngine(copy.deepcopy(base), learning_enabled=True)
    ablation = OWMEngine(copy.deepcopy(base), learning_enabled=False)

    def _bin_brier(eng: OWMEngine) -> float:
        return brier_score([eng.predict(transfer)], gt_bin, stratum="hidden")

    passes = [0]
    engine_pwb = [_per_window_brier(engine, transfer, windows)]
    ablation_pwb = [_per_window_brier(ablation, transfer, windows)]
    engine_bin = [_bin_brier(engine)]
    weight_traj = {s: [_coupling_weight(engine, origin, s)] for s in tracked}

    for i, window in enumerate(windows, 1):
        engine.learn(teach, set(window))
        ablation.learn(teach, set(window))               # no-op: learning disabled
        passes.append(i)
        engine_pwb.append(_per_window_brier(engine, transfer, windows))
        ablation_pwb.append(_per_window_brier(ablation, transfer, windows))
        engine_bin.append(_bin_brier(engine))
        for s in tracked:
            weight_traj[s].append(_coupling_weight(engine, origin, s))

    result = {
        "origin": origin,
        "blast": blast,
        "empirical_rate": rate,
        "passes": passes,
        "engine_per_window_brier": engine_pwb,
        "engine_binary_brier": engine_bin,
        "ablation_per_window_brier": ablation_pwb,
        "weight_trajectories": weight_traj,
    }

    if out_plot is not None:
        _plot(result, out_plot)
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=2))

    return result


def _plot(result: dict, out_plot) -> None:
    """Write the 2-panel convergence figure.

    Top: per-window Brier, engine (learning on) vs flat ablation (learning off).
    Bottom: per-service learned coupling-weight trajectories, each with a dashed
    horizontal line at that service's empirical impact rate (the calibration target).
    """
    passes = result["passes"]
    origin = result["origin"]
    rate = result["empirical_rate"]
    traj = result["weight_trajectories"]

    fig, (ax_brier, ax_weight) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    ax_brier.plot(passes, result["engine_per_window_brier"], "o-",
                  color="#0f766e", label="engine (per-window Brier)")
    ax_brier.plot(passes, result["ablation_per_window_brier"], "s--",
                  color="#94a3b8", label="ablation (learning off)")
    ax_brier.set_ylabel("per-window Brier")
    ax_brier.set_title(f"OWM multi-pass convergence — {origin}")
    ax_brier.legend()
    ax_brier.grid(alpha=0.3)

    for s in sorted(traj):
        color = _COLORS.get(s, _FALLBACK_COLOR)
        ax_weight.plot(passes, traj[s], "o-", color=color, label=f"w[{s}]")
        if s in rate:
            ax_weight.axhline(rate[s], ls=":", color=color, alpha=0.7)  # empirical-rate target
    ax_weight.set_xlabel("learning passes (incident windows folded)")
    ax_weight.set_ylabel("learned coupling weight")
    ax_weight.set_ylim(0, 1.05)
    ax_weight.legend(loc="center right")
    ax_weight.grid(alpha=0.3)

    fig.tight_layout()
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meas", required=True,
                    help="measurement JSON file ({origin, runs}) for one coupling")
    ap.add_argument("--out-plot", default=str(ROOT / "results_live" / "multipass.png"),
                    help="output 2-panel convergence PNG")
    ap.add_argument("--out-json", default=str(ROOT / "results_live" / "multipass.json"),
                    help="output series JSON")
    ap.add_argument("--repo-manifest", default=str(DEFAULT_REPO_MANIFEST), dest="repo_manifest",
                    help="dir with kubernetes-manifests.yaml used to seed the base graph "
                         "(default: the bundled Online Boutique release manifest)")
    args = ap.parse_args()

    meas = json.loads(Path(args.meas).read_text())
    result = run(meas, repo_manifest=args.repo_manifest,
                 out_plot=args.out_plot, out_json=args.out_json)

    osvc = result["origin"].split("::")[0]
    print(f"origin={result['origin']}")
    print(f"denoised binary blast (2/3 GT) = {result['blast']}")
    print(f"empirical impact rates         = "
          f"{ {s: round(r, 3) for s, r in result['empirical_rate'].items()} }\n")
    for i in result["passes"]:
        ws = "  ".join(f"w[{s}]={result['weight_trajectories'][s][i]:.2f}"
                       for s in sorted(result["weight_trajectories"]))
        tag = "(pre)" if i == 0 else "     "
        print(f"pass {i} {tag}: per-window Brier engine={result['engine_per_window_brier'][i]:.4f} "
              f"ablation={result['ablation_per_window_brier'][i]:.4f} "
              f"binary={result['engine_binary_brier'][i]:.4f} | {ws}")
    print("\nfinal learned weights vs empirical rate:")
    final = {s: traj[-1] for s, traj in result["weight_trajectories"].items()}
    for s in sorted(final):
        print(f"  {s:24s} learned={final[s]:.3f}  empirical={result['empirical_rate'].get(s, 0.0):.3f}")
    print(f"\nwrote {args.out_plot} and {args.out_json}")


if __name__ == "__main__":
    main()
