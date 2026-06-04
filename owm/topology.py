"""Assemble a seeded CausalWorldModel from a static parse.

Every propagating edge carries a Beta posterior (alpha, beta); its `weight` is
the posterior mean. The seed prior is Beta(1,1) = mean 0.5: the engine is
genuinely uninformed about how strongly a call propagates failure, and learns
that strength from incidents (T9). Config nodes are added with NO edges.
"""
from __future__ import annotations

from owm.codegraph import ParsedCode
from owm.graph import CausalWorldModel, Kind, Rel, GIVEN

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0


def beta_weight(alpha: float, beta: float) -> float:
    """Posterior mean of Beta(alpha, beta)."""
    return alpha / (alpha + beta)


def beta_variance(alpha: float, beta: float) -> float:
    s = alpha + beta
    return (alpha * beta) / (s * s * (s + 1.0))


def seed_graph(pc: ParsedCode) -> CausalWorldModel:
    cwm = CausalWorldModel()

    for svc in pc.services:
        cwm.add_node(svc, Kind.SERVICE, mitigation=0.0)

    for cfg in pc.config_nodes:
        cwm.add_node(cfg, Kind.CONFIG)

    for owner_file, service in pc.file_owner.items():
        cwm.add_node(owner_file, Kind.FILE)
        # service --contains--> file  (ownership; not a propagating rel)
        cwm.add_edge(service, owner_file, Rel.CONTAINS, weight=1.0, source=GIVEN)

    for svc, rpcs in pc.endpoints.items():
        for rpc in rpcs:
            ep = f"{svc}::{rpc}"
            cwm.add_node(ep, Kind.FILE, endpoint=True)
            cwm.add_edge(svc, ep, Rel.CONTAINS, weight=1.0, source=GIVEN)

    for caller, callee in pc.call_edges:
        # weight = beta_weight(prior); store alpha/beta so learn.py can update it.
        cwm.add_edge(
            caller, callee, Rel.CALLS,
            weight=beta_weight(PRIOR_ALPHA, PRIOR_BETA),
            source=GIVEN, alpha=PRIOR_ALPHA, beta=PRIOR_BETA,
        )

    return cwm
