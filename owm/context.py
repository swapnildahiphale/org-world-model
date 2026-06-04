"""The single context bundle handed to BOTH the engine and the LLM baselines.

Symmetry is the point: whatever the engine learned from (topology + incident
history) is exactly what the LLM is shown. The only knob is include_incidents,
which separates the stateless baseline from the RAG baseline.
"""
from __future__ import annotations

from owm.graph import CausalWorldModel, Rel


def context_pack(cwm: CausalWorldModel, scenario, incidents=None) -> dict:
    call_edges = sorted(
        (u, v) for u, v, d in cwm.edges_view() if d.get("rel") == Rel.CALLS
    )
    return {
        "services": sorted(cwm.services),
        "call_edges": call_edges,
        "change": {
            "touched_services": list(scenario.touched_services),
            "touched_configs": list(scenario.touched_configs),
            "s_t": scenario.s_t,
        },
        "incidents": list(incidents or []),
    }


def facts(pack: dict) -> frozenset[str]:
    fs: set[str] = set(pack["services"])
    fs |= {f"{u}->{v}" for u, v in pack["call_edges"]}
    for inc in pack["incidents"]:
        fs.add(f"incident:{inc['origin']}")
    return frozenset(fs)


def as_llm_text(pack: dict, include_incidents: bool = True) -> str:
    ch = pack["change"]
    lines = [
        "# System topology",
        "Services: " + ", ".join(pack["services"]),
        "",
        "Calls (caller -> callee):",
        *[f"- {u} -> {v}" for u, v in pack["call_edges"]],
        "",
        "# Proposed change",
        f"touched services: {ch['touched_services']}",
        f"touched configs: {ch['touched_configs']}",
        f"cluster state: {ch['s_t']}",
    ]
    if include_incidents and pack["incidents"]:
        lines += ["", "# Incident history"]
        for inc in pack["incidents"]:
            lines.append(f"- change at {inc['origin']} degraded: {', '.join(inc['impacted'])}")
    lines += [
        "",
        "# Task",
        "Predict which of the listed services will be in the blast radius of this change.",
    ]
    return "\n".join(lines)
