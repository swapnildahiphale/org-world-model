"""Deepening: incidents update Beta posteriors and, on surprise, add HIDDEN
COUPLES edges. The learning_enabled flag is the ablation for H2.

A "surprise" = reality impacted a service the engine gave < surprise_threshold.
We add a LEARNED coupling from the change's class-node origin to that service,
initialized Beta(2,1) (the incident is the first positive observation). Edges
that already exist (e.g. the engine already routed it via CALLS) are reweighted,
never duplicated — so learning never silently double-counts a given path.
"""
from __future__ import annotations

from owm.graph import CausalWorldModel, Kind, Rel, LEARNED
from owm.topology import beta_weight

SURPRISE_THRESHOLD = 0.1
NEW_EDGE_ALPHA = 2.0   # prior Beta(1,1) + this incident's one positive observation
NEW_EDGE_BETA = 1.0


def _set_weight(data: dict) -> None:
    data["weight"] = beta_weight(data["alpha"], data["beta"])


def update_edge(cwm: CausalWorldModel, src: str, dst: str, rel: str, *, impacted: bool) -> None:
    data = cwm.g.get_edge_data(src, dst, key=rel)
    if data is None:
        raise KeyError(f"no {rel} edge {src}->{dst}")
    data.setdefault("alpha", 1.0)
    data.setdefault("beta", 1.0)
    if impacted:
        data["alpha"] += 1.0
    else:
        data["beta"] += 1.0
    _set_weight(data)


def observe(cwm: CausalWorldModel, origin: str, impacted_services, predicted_probs,
            *, learning_enabled: bool = True, surprise_threshold: float = SURPRISE_THRESHOLD) -> None:
    """Apply one incident: origin (a class node) caused `impacted_services` to degrade."""
    if not learning_enabled:
        return
    impacted = set(impacted_services)
    if origin not in cwm.g:
        cwm.add_node(origin, Kind.CONFIG)

    for service in cwm.services:
        was_impacted = service in impacted
        has_edge = cwm.g.has_edge(origin, service, key=Rel.COUPLES)
        if has_edge:
            update_edge(cwm, origin, service, Rel.COUPLES, impacted=was_impacted)
        elif was_impacted and predicted_probs.get(service, 0.0) < surprise_threshold:
            # Week-1 simplification: every learned coupling is treated as saturation-sensitive.
            cwm.add_edge(
                origin, service, Rel.COUPLES,
                weight=beta_weight(NEW_EDGE_ALPHA, NEW_EDGE_BETA),
                source=LEARNED, saturation_sensitive=True,
                alpha=NEW_EDGE_ALPHA, beta=NEW_EDGE_BETA,
            )
        # was_impacted but engine already predicted it (>=threshold) and no couples
        # edge => the given topology covered it; do not add a redundant edge.
