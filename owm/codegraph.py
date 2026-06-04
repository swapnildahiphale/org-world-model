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
