"""Provenance-stratified scoring. Headline metrics are reported on the HIDDEN
stratum (Rule 1). Calibration unit = per (scenario, candidate-service).
"""
from __future__ import annotations

from dataclasses import dataclass

from owm.groundtruth import GroundTruth, Prediction

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt


@dataclass
class Bin:
    mean_pred: float
    observed_freq: float
    count: int


def points(preds: list[Prediction], gt: list[GroundTruth]) -> list[tuple[float, int, str]]:
    """Flatten to (predicted_prob, observed_outcome, stratum) over every
    (scenario, candidate-service) pair. Candidates = the prediction's services."""
    gt_by_id = {g.scenario_id: g for g in gt}
    out: list[tuple[float, int, str]] = []
    for p in preds:
        g = gt_by_id.get(p.scenario_id)
        if g is None:
            continue
        for service, prob in p.probs.items():
            outcome = 1 if service in g.impacted else 0
            out.append((float(prob), outcome, g.stratum))
    return out


def _filter(pts, stratum):
    return [(pr, o) for (pr, o, s) in pts if stratum is None or s == stratum]


def brier_score(preds, gt, stratum: str | None = None) -> float:
    pts = _filter(points(preds, gt), stratum)
    if not pts:
        return float("nan")
    return sum((pr - o) ** 2 for pr, o in pts) / len(pts)


def reliability_diagram(preds, gt, n_bins: int = 5, stratum: str | None = None) -> list[Bin]:
    """Equal-frequency bins (robust at small n): sort points by predicted prob,
    split into n_bins roughly-equal groups, report (mean_pred, observed_freq)."""
    pts = sorted(_filter(points(preds, gt), stratum), key=lambda t: t[0])
    if not pts:
        return []
    n = len(pts)
    bins: list[Bin] = []
    for i in range(n_bins):
        lo = (i * n) // n_bins
        hi = ((i + 1) * n) // n_bins
        chunk = pts[lo:hi]
        if not chunk:
            continue
        mp = sum(pr for pr, _ in chunk) / len(chunk)
        of = sum(o for _, o in chunk) / len(chunk)
        bins.append(Bin(mean_pred=mp, observed_freq=of, count=len(chunk)))
    return bins


def deepening(pre_preds, post_preds, gt, stratum: str = "hidden") -> dict:
    """2-point deepening on the HIDDEN stratum: Brier before vs after learning.
    A real deepening has brier_post < brier_pre on held-out TRANSFER changes."""
    return {
        "brier_pre": brier_score(pre_preds, gt, stratum=stratum),
        "brier_post": brier_score(post_preds, gt, stratum=stratum),
    }


def state_delta(healthy: Prediction, saturated: Prediction) -> dict[str, float]:
    """Per-service increase in predicted blast probability from healthy -> saturated
    for the SAME change. Positive on saturation-sensitive services = the money shot."""
    keys = set(healthy.probs) & set(saturated.probs)
    return {s: saturated.probs[s] - healthy.probs[s] for s in keys}


def plot_reliability(bins: list[Bin], path) -> None:
    """The primary calibration artifact: mean predicted prob vs observed frequency,
    with the y=x perfect-calibration diagonal."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="perfect calibration")
    if bins:
        ax.plot([b.mean_pred for b in bins], [b.observed_freq for b in bins],
                "o-", color="#0f766e", label="OWM engine")
    ax.set_xlabel("mean predicted P(impact)")
    ax.set_ylabel("observed fault frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
