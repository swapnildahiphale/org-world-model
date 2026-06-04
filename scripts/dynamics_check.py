"""[SYNTHETIC mechanism-validation] — exercise the OWM learner against a known
controlled truth and report whether the Beta-update dynamics behave.

This is a *mechanism* check, deliberately separate from the honest headline
(the HIDDEN-stratum Brier produced by scripts/run.py). It answers three narrow
questions about the learning machinery, with no claim about real-world accuracy:

  1. CONVERGENCE  — folding incidents from a coupling with true rate ``p`` drives
     the learned COUPLES weight toward ``p`` (exact for a deterministic p=1.0
     coupling; trending for stochastic ones).
  2. SPURIOUS     — with coincidental co-degradation noise, the learner will add
     LEARNED edges to UNcoupled services. We measure how many (the precision
     risk; expected > 0). This is reported as a risk, not hidden.
  3. ORDER-INVARIANCE — the same multiset of incidents folded in two different
     orders yields the same per-edge Beta posterior (alpha, beta), as it must
     for a conjugate count-based update.

Everything is built on the real ``OWMEngine`` / ``observe`` path so the harness
validates the shipping mechanism, not a re-implementation. No network/cluster/LLM.

The truth lives in :mod:`groundtruth.oracle`. We seed a base graph from Online
Boutique's real topology when a manifest is discoverable, otherwise a caller can
inject a ``cwm`` for testability (the mechanism is identical either way).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Callable, Mapping

from owm.codegraph import parse_repo
from owm.engine import OWMEngine
from owm.graph import CausalWorldModel, Kind, Rel, LEARNED
from owm.scenarios import Scenario
from owm.topology import seed_graph

from groundtruth.oracle import TrueCoupling, stream

ROOT = Path(__file__).resolve().parents[1]
# Candidate manifests for a base over OB's real topology (first that exists wins).
_MANIFEST_DIRS = (
    ROOT / "data" / "online_boutique" / "repo" / "release",
    ROOT / "tests" / "fixtures" / "ob_mini",
)
DEFAULT_ORIGIN = "synthetic::ORACLE_KNOB"


# --------------------------------------------------------------------------- #
# Base construction
# --------------------------------------------------------------------------- #
def _find_manifest_dir() -> Path | None:
    for d in _MANIFEST_DIRS:
        if (d / "kubernetes-manifests.yaml").is_file():
            return d
    return None


def base_cwm(*, repo_path: Path | None = None) -> CausalWorldModel:
    """Seed a base CausalWorldModel from OB's real topology.

    Uses ``repo_path`` if given, else the first discoverable bundled manifest.
    Raises if none is available — callers wanting full isolation should pass a
    ``cwm_factory`` to the check functions instead of relying on this.
    """
    repo_path = repo_path or _find_manifest_dir()
    if repo_path is None:
        raise FileNotFoundError(
            "no kubernetes-manifests.yaml found for a base; "
            "inject a cwm_factory for isolated runs"
        )
    return seed_graph(parse_repo(repo_path))


def make_factory(
    origin: str = DEFAULT_ORIGIN,
    *,
    repo_path: Path | None = None,
    cwm: CausalWorldModel | None = None,
) -> Callable[[], CausalWorldModel]:
    """Return a zero-arg factory producing a FRESH base each call.

    Each fresh graph has the synthetic ``origin`` config node added (unconnected,
    exactly like a parsed-but-uncoupled knob — the surface a hidden coupling hides
    behind). If ``cwm`` is supplied it is used as the template (deep-copied per
    call); otherwise the base is seeded from the topology via :func:`base_cwm`.
    """
    template = cwm if cwm is not None else base_cwm(repo_path=repo_path)

    def factory() -> CausalWorldModel:
        fresh = copy.deepcopy(template)
        if origin not in fresh.g:
            fresh.add_node(origin, Kind.CONFIG)
        return fresh

    return factory


# --------------------------------------------------------------------------- #
# Folding incidents through the real engine
# --------------------------------------------------------------------------- #
def _teach_scenario(origin: str) -> Scenario:
    return Scenario(id=f"oracle-teach::{origin}", stratum="hidden",
                    role="teach", touched_configs=(origin,), s_t="healthy")


def fold_incidents(cwm: CausalWorldModel, origin: str, incidents) -> OWMEngine:
    """Fold a sequence of impacted-sets into a fresh learning engine over ``cwm``.

    Returns the engine (its ``cwm`` now carries the learned posteriors).
    """
    engine = OWMEngine(cwm, learning_enabled=True)
    teach = _teach_scenario(origin)
    for impacted in incidents:
        engine.learn(teach, set(impacted))
    return engine


def learned_couples(cwm: CausalWorldModel, origin: str) -> dict[str, dict]:
    """Per-service LEARNED COUPLES edge data out of ``origin`` -> {service: data}."""
    out: dict[str, dict] = {}
    if origin not in cwm.g:
        return out
    for _src, dst, key, data in cwm.g.out_edges(origin, keys=True, data=True):
        if key == Rel.COUPLES and data.get("source") == LEARNED:
            out[dst] = data
    return out


# --------------------------------------------------------------------------- #
# Check 1: convergence
# --------------------------------------------------------------------------- #
def check_convergence(
    cwm_factory: Callable[[], CausalWorldModel],
    origin: str,
    true_probs: Mapping[str, float],
    n: int,
    seed: int,
    *,
    tol: float | None = None,
) -> dict:
    """Fold ``n`` sampled incidents and report the learned weight per service.

    For each service with a deterministic true coupling (``p == 1.0``) we assert
    the learned weight is within ``tol`` of 1.0. For stochastic couplings we only
    report (and flag) the trend toward ``p`` — a short stream is not expected to
    pin it exactly. Returns a dict of measured quantities.

    Honest tolerance: a deterministic coupling cannot reach exactly 1.0 in finite
    n. The edge is born Beta(2,1) on its first surprising impact and each later
    impact adds 1 to alpha, so after ``n`` all-positive observations the posterior
    mean is ``(1+n)/(2+n)`` — a gap of ``1/(n+2)`` below 1.0. When ``tol`` is None
    we set the pass band to that exact asymptotic gap (plus a tiny epsilon) so the
    criterion tracks the mechanism's real convergence rate rather than an arbitrary
    constant. Pass an explicit ``tol`` to override.
    """
    cwm = cwm_factory()
    services = cwm.services
    rng = random.Random(seed)
    incidents = stream(origin, true_probs, n, rng, noise=0.0, all_services=services)

    engine = fold_incidents(cwm, origin, incidents)
    edges = learned_couples(engine.cwm, origin)

    # asymptotic gap to 1.0 for n all-positive observations on a born-Beta(2,1) edge
    det_tol = tol if tol is not None else (1.0 / (n + 2.0) + 1e-9)

    weights: dict[str, float] = {}
    sources: dict[str, str] = {}
    converged: dict[str, bool] = {}
    for service, p in true_probs.items():
        data = edges.get(service)
        w = float(data["weight"]) if data else 0.0
        weights[service] = w
        sources[service] = (data.get("source") if data else None)
        if p >= 1.0 - 1e-9:
            ok = abs(w - p) <= det_tol
            converged[service] = ok
            assert ok, (
                f"[SYNTHETIC] deterministic coupling {origin}->{service} (p={p}) "
                f"did not converge: learned weight {w:.4f} not within {det_tol:.4f} "
                f"of {p} after {n} incidents"
            )
            assert sources[service] == LEARNED, (
                f"[SYNTHETIC] edge {origin}->{service} not marked LEARNED"
            )
        else:
            # Stochastic: report closeness; trending = learned weight within a
            # looser band of p. Not asserted (a short stream won't pin it exactly).
            converged[service] = abs(w - p) <= max(det_tol, 0.25)

    return {
        "origin": origin,
        "n": n,
        "seed": seed,
        "true_probs": dict(true_probs),
        "learned_weights": weights,
        "sources": sources,
        "converged": converged,
        "convergence_tol": det_tol,
        "n_services": len(services),
    }


# --------------------------------------------------------------------------- #
# Check 2: spurious edges (precision risk)
# --------------------------------------------------------------------------- #
def check_spurious(
    cwm_factory: Callable[[], CausalWorldModel],
    origin: str,
    true_probs: Mapping[str, float],
    n: int,
    seed: int,
    *,
    noise: float = 0.3,
) -> dict:
    """Run with co-degradation ``noise`` and count LEARNED edges to services NOT
    in ``true_probs`` (the spurious couplings). Reports count + rate; expected > 0.
    This surfaces the precision risk honestly rather than hiding it.
    """
    cwm = cwm_factory()
    services = cwm.services
    rng = random.Random(seed)
    incidents = stream(origin, true_probs, n, rng, noise=noise, all_services=services)

    engine = fold_incidents(cwm, origin, incidents)
    edges = learned_couples(engine.cwm, origin)

    spurious = sorted(s for s in edges if s not in true_probs)
    n_uncoupled = len([s for s in services if s not in true_probs])
    return {
        "origin": origin,
        "n": n,
        "seed": seed,
        "noise": noise,
        "spurious_edges": spurious,
        "n_spurious": len(spurious),
        "n_uncoupled_services": n_uncoupled,
        "spurious_rate": (len(spurious) / n_uncoupled) if n_uncoupled else 0.0,
        "true_edges_learned": sorted(s for s in edges if s in true_probs),
    }


# --------------------------------------------------------------------------- #
# Check 3: order-invariance
# --------------------------------------------------------------------------- #
def check_order_invariance(
    cwm_factory: Callable[[], CausalWorldModel],
    origin: str,
    true_probs: Mapping[str, float],
    n: int,
    seed: int,
    *,
    tol: float = 1e-9,
) -> dict:
    """Fold the SAME multiset of incidents in two orders into two fresh engines;
    assert the final per-edge (alpha, beta) match within ``tol``.

    The Beta update is a pure count of impacted/not-impacted observations, so the
    posterior must be order-independent — but only once an edge EXISTS. A learned
    edge is created on the first *surprising* impact; thereafter all observations
    (including the creating one) are counted. To make the multiset folding genuinely
    order-invariant we fold the identical list, just permuted, so the same incidents
    contribute the same counts regardless of position.
    """
    cwm_a = cwm_factory()
    services = cwm_a.services
    rng = random.Random(seed)
    incidents = stream(origin, true_probs, n, rng, noise=0.0, all_services=services)

    forward = list(incidents)
    reverse = list(reversed(incidents))

    eng_a = fold_incidents(cwm_a, origin, forward)
    eng_b = fold_incidents(cwm_factory(), origin, reverse)

    edges_a = learned_couples(eng_a.cwm, origin)
    edges_b = learned_couples(eng_b.cwm, origin)

    keys = sorted(set(edges_a) | set(edges_b))
    per_edge = {}
    invariant = True
    for k in keys:
        a = edges_a.get(k, {})
        b = edges_b.get(k, {})
        aa, ab = float(a.get("alpha", 0.0)), float(a.get("beta", 0.0))
        ba, bb = float(b.get("alpha", 0.0)), float(b.get("beta", 0.0))
        match = abs(aa - ba) <= tol and abs(ab - bb) <= tol and (k in edges_a) == (k in edges_b)
        per_edge[k] = {
            "forward": {"alpha": aa, "beta": ab},
            "reverse": {"alpha": ba, "beta": bb},
            "match": match,
        }
        invariant = invariant and match

    assert invariant, (
        f"[SYNTHETIC] order-invariance violated for {origin}: per-edge posteriors "
        f"differ between fold orders: {per_edge}"
    )
    return {
        "origin": origin,
        "n": n,
        "seed": seed,
        "per_edge": per_edge,
        "order_invariant": invariant,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description="[SYNTHETIC mechanism-validation] OWM learning-dynamics checks"
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=20, help="incidents per check")
    ap.add_argument("--origin", default=DEFAULT_ORIGIN)
    ap.add_argument("--json", action="store_true", help="print measured dict as JSON")
    args = ap.parse_args(argv)

    origin = args.origin
    factory = make_factory(origin)
    services = factory().services

    # A small synthetic truth over real service names: one deterministic coupling,
    # one stochastic. Pick stable, present services.
    det = services[0]
    stoch = services[1] if len(services) > 1 else services[0]
    true_probs = {det: 1.0, stoch: 0.6}

    conv = check_convergence(factory, origin, true_probs, args.n, args.seed)
    spur = check_spurious(factory, origin, true_probs, args.n, args.seed, noise=0.3)
    order = check_order_invariance(factory, origin, true_probs, args.n, args.seed)

    measured = {
        "label": "SYNTHETIC mechanism-validation (NOT a headline metric)",
        "origin": origin,
        "n": args.n,
        "seed": args.seed,
        "n_services": len(services),
        "convergence": conv,
        "spurious": spur,
        "order_invariance": order,
    }

    prefix = "[SYNTHETIC mechanism-validation]"
    print(f"{prefix} OWM learning-dynamics check "
          f"(origin={origin}, n={args.n}, seed={args.seed}, services={len(services)})")
    print(f"{prefix} 1. CONVERGENCE (true -> learned weight):")
    for svc, p in true_probs.items():
        w = conv["learned_weights"].get(svc, 0.0)
        tag = "deterministic" if p >= 1.0 - 1e-9 else "stochastic "
        ok = "OK" if conv["converged"].get(svc) else "TREND"
        print(f"{prefix}      {tag} {svc}: true={p:.2f} learned={w:.3f}  [{ok}]")
    print(f"{prefix} 2. SPURIOUS edges under noise={spur['noise']}: "
          f"{spur['n_spurious']} of {spur['n_uncoupled_services']} uncoupled "
          f"(rate={spur['spurious_rate']:.2f}) -> {spur['spurious_edges']} "
          f"[precision risk, expected > 0]")
    print(f"{prefix} 3. ORDER-INVARIANCE: {'HOLDS' if order['order_invariant'] else 'VIOLATED'} "
          f"({len(order['per_edge'])} learned edge(s) compared)")
    print(f"{prefix} reminder: synthetic mechanism check only — the honest headline "
          f"is the HIDDEN-stratum Brier from scripts/run.py.")

    if args.json:
        print(json.dumps(measured, indent=2, default=str))
    return measured


if __name__ == "__main__":
    main()
