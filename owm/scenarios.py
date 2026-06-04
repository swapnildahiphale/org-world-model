"""Scenario schema + the canonical Week-1 scenario set.

A Scenario is a change to predict, plus the provenance/role labels the eval
needs.

The learned-coupling origin is the genuinely-parsed config node (e.g.
"productcatalogservice::EXTRA_LATENCY") — the node the static parser emits but
leaves unconnected (a latency knob's downstream blast is not statically knowable).
TEACH and TRANSFER are DIFFERENT changes (different scenario ids) touching the SAME
parsed knob, so transfer is genuine generalization, not replay. NEGATIVE touches a
DIFFERENT parsed knob on the same service (a benign sibling), so the engine's
non-firing there shows the lesson stays LOCAL to the knob that caused the incident.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

LATENCY_KNOB = "productcatalogservice::EXTRA_LATENCY"
BENIGN_KNOB = "productcatalogservice::DISABLE_PROFILER"
HIDDEN_EDGE = "pc-latency-coupling"


@dataclass(frozen=True)
class Scenario:
    id: str
    stratum: str                       # "given" | "hidden"
    role: str | None = None            # "teach" | "transfer" | "negative" | None
    hidden_edge: str | None = None
    touched_services: tuple[str, ...] = ()
    touched_configs: tuple[str, ...] = ()
    s_t: str = "healthy"

    def __post_init__(self):
        # Coerce list inputs to tuples so Scenario(touched_configs=[...]) == from_dict(...)
        object.__setattr__(self, "touched_services", tuple(self.touched_services))
        object.__setattr__(self, "touched_configs", tuple(self.touched_configs))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["touched_services"] = list(self.touched_services)
        d["touched_configs"] = list(self.touched_configs)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(
            id=d["id"], stratum=d["stratum"], role=d.get("role"),
            hidden_edge=d.get("hidden_edge"),
            touched_services=tuple(d.get("touched_services") or ()),
            touched_configs=tuple(d.get("touched_configs") or ()),
            s_t=d.get("s_t", "healthy"),
        )


def default_scenarios() -> list[Scenario]:
    return [
        # GIVEN floor: a code change to a service the engine was handed.
        Scenario(id="given-pc-handler", stratum="given",
                 touched_services=("productcatalogservice",), s_t="healthy"),
        # HIDDEN family — all config-class changes; latency class is the hidden edge.
        Scenario(id="teach-latency-1", stratum="hidden", role="teach",
                 hidden_edge=HIDDEN_EDGE, touched_configs=(LATENCY_KNOB,), s_t="healthy"),
        Scenario(id="transfer-latency-1", stratum="hidden", role="transfer",
                 hidden_edge=HIDDEN_EDGE, touched_configs=(LATENCY_KNOB,), s_t="healthy"),
        Scenario(id="negative-flag-1", stratum="hidden", role="negative",
                 hidden_edge=None, touched_configs=(BENIGN_KNOB,), s_t="healthy"),
    ]


def _canonical(scenarios: list[Scenario]) -> str:
    rows = sorted((s.to_dict() for s in scenarios), key=lambda d: d["id"])
    return json.dumps(rows, sort_keys=True, indent=2)


def emit(scenarios: list[Scenario], path) -> None:
    Path(path).write_text(_canonical(scenarios))


def content_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
