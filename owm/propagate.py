"""do(change, state): noisy-OR forward propagation of failure over the graph.

Fault flows cause->effect. Per relation:
  - CALLS / DEPENDS_ON: a service's fault-parents are its SUCCESSORS (callees),
    because A--calls-->B means B breaking risks A.
  - COUPLES: a service's fault-parent is its PREDECESSOR (the config), because
    C--couples-->S means C changing risks S.
Mitigating controls on the dependent lower incoming impact; saturated state
collapses mitigation and amplifies saturation-sensitive edges.
"""
from __future__ import annotations

from owm.graph import CausalWorldModel, Kind, Rel

MAX_ITERS = 50
EPS = 1e-6
SAT_MIT_FACTOR = 0.2   # under saturation, mitigation retains only 20% of its effect
SAT_AMP = 0.5          # saturation-sensitive edge weight moves halfway toward 1.0


def _origins(cwm, touched_services, touched_configs) -> dict[str, float]:
    origins: dict[str, float] = {}
    for s in touched_services or []:
        origins[s] = 1.0
    for c in touched_configs or []:
        origins[c] = 1.0
    return origins


def _fault_parents(cwm: CausalWorldModel, effect: str):
    """Yield (cause, base_weight, saturation_sensitive) flowing INTO `effect`."""
    g = cwm.g
    # CALLS / DEPENDS_ON: parents are this node's callees/datastores (successors)
    for _e, callee, key, data in g.out_edges(effect, keys=True, data=True):
        if key in (Rel.CALLS, Rel.DEPENDS_ON):
            yield callee, float(data.get("weight", 1.0)), bool(data.get("saturation_sensitive", False))
    # COUPLES: parents are configs that couple to this node (predecessors)
    for cfg, _e, key, data in g.in_edges(effect, keys=True, data=True):
        if key == Rel.COUPLES:
            yield cfg, float(data.get("weight", 1.0)), bool(data.get("saturation_sensitive", False))


def _effective_weight(cwm, effect, base, sat_sensitive, s_t) -> float:
    mit = float(cwm.g.nodes[effect].get("mitigation", 0.0))
    if s_t == "saturated":
        mit *= SAT_MIT_FACTOR
        if sat_sensitive:
            base = base + (1.0 - base) * SAT_AMP
    return base * (1.0 - mit)


def predict_blast(cwm: CausalWorldModel, *, touched_services=None, touched_configs=None,
                  s_t: str = "healthy") -> dict[str, float]:
    """Return P(impact) per service (excluding perturbed origin services)."""
    origins = _origins(cwm, touched_services, touched_configs)
    services = cwm.services
    impact = {n: (1.0 if n in origins else 0.0) for n in cwm.g.nodes}

    for _ in range(MAX_ITERS):
        nxt = dict(impact)
        max_delta = 0.0
        for effect in services:
            if effect in origins:
                continue
            prod = 1.0
            for cause, base, sat in _fault_parents(cwm, effect):
                w = _effective_weight(cwm, effect, base, sat, s_t)
                prod *= (1.0 - impact[cause] * w)
            val = 1.0 - prod
            max_delta = max(max_delta, abs(val - impact[effect]))
            nxt[effect] = val
        impact = nxt
        if max_delta < EPS:
            break

    return {s: impact[s] for s in services if s not in origins}
