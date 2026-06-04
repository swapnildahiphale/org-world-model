"""Static parse of an Online Boutique checkout into the GIVEN seed structure.

Deliberately language-agnostic: the inter-service topology comes from the
`*_SERVICE_ADDR` env wiring in the k8s manifests (uniform across all 5 langs),
ownership comes from the `src/<service>/` directory layout, and config knobs
(env vars that are NOT host:port) become isolated `config` nodes — the surface
where genuinely-hidden couplings live. We never infer a coupling's effect here;
that is exactly what static analysis cannot know (Rule 2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# value looks like "servicename:1234"
_ADDR_RE = re.compile(r"^(?P<host>[a-z0-9-]+):(?P<port>\d+)$")
# a Go/any rpc handler: capture method name after "func ... <Name>("
_RPC_RE = re.compile(r"\brpc\s+(\w+)\s*\(")


@dataclass
class ParsedCode:
    services: list[str] = field(default_factory=list)
    call_edges: list[tuple[str, str]] = field(default_factory=list)   # (caller, callee)
    config_nodes: list[str] = field(default_factory=list)             # "<svc>::<ENV>"
    file_owner: dict[str, str] = field(default_factory=dict)          # relpath -> service
    endpoints: dict[str, list[str]] = field(default_factory=dict)     # service -> [rpc]


def _load_deployments(manifest: Path) -> list[dict]:
    docs = [d for d in yaml.safe_load_all(manifest.read_text()) if d]
    return [d for d in docs if d.get("kind") == "Deployment"]


def _envs(dep: dict) -> list[dict]:
    containers = dep["spec"]["template"]["spec"]["containers"]
    out: list[dict] = []
    for c in containers:
        out.extend(c.get("env", []) or [])
    return out


def _containers(dep: dict) -> list[dict]:
    spec = dep.get("spec") or {}
    template = spec.get("template") or {}
    tspec = template.get("spec") or {}
    return tspec.get("containers") or []


def _resource_knobs(dep: dict) -> list[str]:
    """Genuine scale/resource config knobs declared in the manifest.

    These are real surfaces a change can touch, but — like any config knob —
    their downstream blast is NOT statically knowable, so we emit them as inert
    `<KNOB>` nodes and deliberately create no edges (Rule 2). Returns the knob
    suffixes (e.g. "REPLICAS", "LIMITS_CPU") in a stable, deduped order.
    """
    knobs: list[str] = []
    spec = dep.get("spec") or {}
    if spec.get("replicas") is not None:
        knobs.append("REPLICAS")

    # resources.{limits,requests}.{cpu,memory} on any container
    pairs = (
        ("limits", "cpu", "LIMITS_CPU"),
        ("limits", "memory", "LIMITS_MEMORY"),
        ("requests", "cpu", "REQUESTS_CPU"),
        ("requests", "memory", "REQUESTS_MEMORY"),
    )
    for c in _containers(dep):
        resources = c.get("resources") or {}
        for bucket, key, knob in pairs:
            if (resources.get(bucket) or {}).get(key) is not None and knob not in knobs:
                knobs.append(knob)
    return knobs


def parse_repo(root: Path) -> ParsedCode:
    root = Path(root)
    pc = ParsedCode()

    # --- services + call edges + config nodes (from manifests) ---
    manifest = root / "kubernetes-manifests.yaml"
    for dep in _load_deployments(manifest):
        svc = dep["metadata"]["name"]
        pc.services.append(svc)
        for env in _envs(dep):
            name, value = env.get("name", ""), str(env.get("value", ""))
            m = _ADDR_RE.match(value)
            if m:
                pc.call_edges.append((svc, m.group("host")))
            else:
                pc.config_nodes.append(f"{svc}::{name}")
        # scale/resource knobs that genuinely exist in the manifest become
        # inert config nodes too (no edges — their blast radius is learned, Rule 2).
        for knob in _resource_knobs(dep):
            node = f"{svc}::{knob}"
            if node not in pc.config_nodes:
                pc.config_nodes.append(node)
    pc.services.sort()

    # --- ownership (from src/<service>/) ---
    src = root / "src"
    if src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file():
                rel = f.relative_to(root).as_posix()
                # src/<service>/...
                service = f.relative_to(src).parts[0]
                pc.file_owner[rel] = service

    # --- endpoints (productcatalog showcase) ---
    proto = root / "protos" / "demo.proto"
    if proto.is_file():
        names = _RPC_RE.findall(proto.read_text())
        if names:
            pc.endpoints["productcatalogservice"] = names

    return pc
