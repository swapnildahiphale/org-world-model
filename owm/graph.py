"""The join graph — schema for the causal world model.

One graph links the layers a senior engineer reasons over but no tool fuses:

    commit --touches--> file --(belongs to)--> service --calls--> service
       \\                                                          ^
        \\--root_cause-- incident --caused--> service ------------/

Nodes carry a ``kind``; edges carry a ``rel``, a ``weight`` (a propagation
probability in [0,1]), and a ``source`` ("given" = seeded from known structure,
"learned" = added/reweighted from an observed incident). The learned/given split
is what lets the model *deepen*: incidents add edges the static structure missed.

Deliberately built on networkx so the spike stays dependency-light and the
mechanism (weighted forward propagation over a typed graph) is fully legible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import networkx as nx


# --- node kinds -------------------------------------------------------------
class Kind:
    SERVICE = "service"
    DATASTORE = "datastore"
    FILE = "file"
    COMMIT = "commit"
    INCIDENT = "incident"
    CONFIG = "config"          # a config knob (env var/flag) a change can touch


# --- edge relations ---------------------------------------------------------
class Rel:
    CALLS = "calls"            # service -> service (caller -> callee)
    DEPENDS_ON = "depends_on"  # service -> datastore
    CONTAINS = "contains"      # service -> file  (ownership)
    TOUCHES = "touches"        # commit -> file
    AFFECTS = "affects"        # commit -> service (derived)
    CAUSED = "caused"          # incident -> service
    ROOT_CAUSE = "root_cause"  # incident -> commit
    COUPLES = "couples"        # config -> service: a learned runtime coupling


# NOTE: Rel.COUPLES is intentionally absent FROM PROPAGATING_RELS — it flows cause->effect and is
# traversed forward by propagate.py, not backward via callers_of().
#: Relations a fault propagates *backward* along: if a callee/dependency breaks,
#: its callers are at risk. Blast radius = predecessors over these relations.
PROPAGATING_RELS = frozenset({Rel.CALLS, Rel.DEPENDS_ON})

#: Source provenance for an edge.
GIVEN = "given"
LEARNED = "learned"


@dataclass
class CausalWorldModel:
    """A typed, weighted causal graph over an org's systems.

    Thin wrapper around ``nx.MultiDiGraph`` with the vocabulary of the join and
    the few queries the spike needs. Multi-graph because two nodes can be linked
    by more than one relation (e.g. a service both ``calls`` another and shares
    a ``learned`` edge to it).
    """

    g: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    # -- construction --------------------------------------------------------
    def add_node(self, node_id: str, kind: str, **attrs) -> str:
        self.g.add_node(node_id, kind=kind, **attrs)
        return node_id

    def add_edge(
        self,
        src: str,
        dst: str,
        rel: str,
        weight: float = 1.0,
        source: str = GIVEN,
        **meta,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0,1], got {weight!r}")
        # key by relation so re-adding the same relation updates in place
        self.g.add_edge(src, dst, key=rel, rel=rel, weight=weight, source=source, **meta)

    # -- accessors -----------------------------------------------------------
    def nodes_of(self, kind: str) -> list[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("kind") == kind]

    @property
    def services(self) -> list[str]:
        return self.nodes_of(Kind.SERVICE)

    @property
    def datastores(self) -> list[str]:
        return self.nodes_of(Kind.DATASTORE)

    @property
    def files(self) -> list[str]:
        return self.nodes_of(Kind.FILE)

    def has_edge(self, src: str, dst: str, rel: str) -> bool:
        return self.g.has_edge(src, dst, key=rel)

    def owning_service(self, file_id: str) -> str | None:
        """The service that ``contains`` this file (the file's home)."""
        for s, _f, key in self.g.in_edges(file_id, keys=True):
            if key == Rel.CONTAINS:
                return s
        return None

    def callers_of(self, node: str) -> Iterator[tuple[str, float]]:
        """Direct upstream dependents: who is at risk if ``node`` breaks.

        Yields ``(caller, weight)`` over propagating relations (reverse direction:
        an edge ``A --calls--> B`` means A depends on B, so B breaking risks A).
        """
        for caller, _callee, key, data in self.g.in_edges(node, keys=True, data=True):
            if key in PROPAGATING_RELS:
                yield caller, float(data.get("weight", 1.0))

    def edges_view(self, source: str | None = None) -> Iterable[tuple[str, str, dict]]:
        for u, v, data in self.g.edges(data=True):
            if source is None or data.get("source") == source:
                yield u, v, data

    # -- summary -------------------------------------------------------------
    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for _n, d in self.g.nodes(data=True):
            kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
        rels: dict[str, int] = {}
        learned = 0
        for _u, _v, d in self.g.edges(data=True):
            rels[d.get("rel", "?")] = rels.get(d.get("rel", "?"), 0) + 1
            if d.get("source") == LEARNED:
                learned += 1
        lines = [
            f"CausalWorldModel: {self.g.number_of_nodes()} nodes, "
            f"{self.g.number_of_edges()} edges ({learned} learned)",
            "  nodes by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
            "  edges by rel:  " + ", ".join(f"{k}={v}" for k, v in sorted(rels.items())),
        ]
        return "\n".join(lines)
