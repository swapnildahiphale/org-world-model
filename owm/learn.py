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

__all__ = ["observe", "update_edge"]

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
            *, learning_enabled: bool = True, surprise_threshold: float = SURPRISE_THRESHOLD,
            learn_call_edges: bool = False) -> None:
    """Apply one incident: origin (a class node) caused `impacted_services` to degrade.

    By default (``learn_call_edges=False``) this updates/adds only COUPLES edges,
    exactly as in Week-1 — the seeded CALLS edges (Beta(1,1) from ``seed_graph``)
    are left untouched, which is the long-standing gap.

    Week-2 heuristic (opt-in, ``learn_call_edges=True``)
    ---------------------------------------------------
    AFTER the COUPLES pass, also Beta-update the seeded CALLS edges with a simple,
    local credit rule. For every CALLS edge ``caller -> callee`` we only consider
    it as carrying signal when its **callee was in the observed blast**; if the
    callee was not impacted, the edge tells us nothing this incident and is left
    alone. When the callee *was* impacted:

      * ``caller`` also impacted  -> ``update_edge(..., Rel.CALLS, impacted=True)``
        The fault propagating *along this call* is consistent with the observed
        blast (callee broke and so did its caller), so this edge gets a positive
        observation: alpha += 1, posterior mean (weight) rises toward 1.
      * ``caller`` NOT impacted    -> ``update_edge(..., Rel.CALLS, impacted=False)``
        The callee broke but its caller did *not* go down, i.e. failure did not
        propagate across this edge this time — evidence the edge is weak: beta += 1,
        posterior mean (weight) falls toward 0.

    This is deliberately a per-edge *local* rule (no path attribution, no
    saturation model) — defensible as a first-order credit assignment and easy to
    audit. It is strictly additive: it never creates, removes, or reorders edges,
    and runs only when explicitly enabled, so the default behavior is byte-for-byte
    unchanged for every existing caller.
    """
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

    if learn_call_edges:
        # Week-2 opt-in: Beta-update seeded CALLS edges via the local credit rule
        # documented above. Collect the matching edges first so we never mutate the
        # MultiDiGraph while iterating over it.
        call_edges = [
            (caller, callee)
            for caller, callee, key in cwm.g.edges(keys=True)
            if key == Rel.CALLS
        ]
        for caller, callee in call_edges:
            if callee not in impacted:
                continue  # callee was not in the blast -> no signal for this edge
            update_edge(cwm, caller, callee, Rel.CALLS, impacted=caller in impacted)
