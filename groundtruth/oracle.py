"""A small controlled-truth ORACLE for validating the learning MECHANISM.

This is *synthetic* mechanism-validation, NOT a headline metric. It posits a
known, hand-set "true coupling" from a config knob to a set of services, samples
incidents from it, and lets the rest of the harness (scripts/dynamics_check.py)
fold those incidents into a real ``OWMEngine`` to check that the learner recovers
the truth it was never told.

Nothing here touches a network, a cluster, or an LLM. Determinism comes entirely
from an injected ``random.Random`` instance — the same seed yields the same
incident stream, so every reported number is reproducible.

A ``TrueCoupling`` is::

    {origin_config_node: {service: true_prob, ...}, ...}

i.e. for each origin, the per-service probability that a change to that origin
degrades the service. ``true_prob == 1.0`` is a deterministic coupling (every
incident hits it); ``0 < p < 1`` is a stochastic one (hits at rate ``p``).
"""
from __future__ import annotations

import random
from typing import Iterable, Mapping

# A TrueCoupling: origin -> {service -> true_prob in [0,1]}
TrueCoupling = dict[str, dict[str, float]]


def sample_incident(
    origin: str,
    true_probs: Mapping[str, float],
    rng: random.Random,
    *,
    noise: float = 0.0,
    all_services: Iterable[str],
) -> set[str]:
    """Sample ONE incident's impacted-service set from a true coupling.

    Each service in ``true_probs`` is impacted independently with its own
    ``true_prob``. With probability ``noise`` we additionally impact one
    randomly-chosen service that is NOT coupled to ``origin`` — modelling
    coincidental co-degradation (the noise an over-eager learner will mistake
    for a real edge, i.e. the precision risk).

    ``origin`` is accepted for symmetry / readability of call sites; the impacted
    set is determined by ``true_probs`` and ``noise`` alone. Fully deterministic
    given ``rng``.
    """
    all_services = list(all_services)
    impacted: set[str] = set()
    # Coupled services: independent Bernoulli draws at their true rates.
    # Iterate in a stable (sorted) order so the rng is consumed deterministically.
    for service in sorted(true_probs):
        p = true_probs[service]
        if rng.random() < p:
            impacted.add(service)

    # Coincidental co-degradation: occasionally an UNcoupled service also dips.
    if noise > 0.0 and rng.random() < noise:
        uncoupled = [s for s in all_services if s not in true_probs]
        if uncoupled:
            impacted.add(rng.choice(sorted(uncoupled)))

    return impacted


def stream(
    origin: str,
    true_probs: Mapping[str, float],
    n: int,
    rng: random.Random,
    **kw,
) -> list[set[str]]:
    """Sample ``n`` incidents as a list of impacted-service sets.

    Deterministic given ``rng``. Extra keyword args (``noise``, ``all_services``)
    are forwarded to :func:`sample_incident`.
    """
    return [sample_incident(origin, true_probs, rng, **kw) for _ in range(n)]
