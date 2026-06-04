from pathlib import Path
from owm.codegraph import parse_repo, ParsedCode

FIXTURE = Path(__file__).parent / "fixtures" / "ob_mini"


def parsed() -> ParsedCode:
    return parse_repo(FIXTURE)


def test_discovers_all_services():
    assert set(parsed().services) == {
        "frontend", "recommendationservice", "productcatalogservice", "currencyservice",
    }


def test_call_edges_from_service_addr_env():
    edges = set(parsed().call_edges)
    assert ("frontend", "productcatalogservice") in edges
    assert ("frontend", "currencyservice") in edges
    assert ("frontend", "recommendationservice") in edges
    assert ("recommendationservice", "productcatalogservice") in edges


def test_non_addr_env_becomes_config_node_not_an_edge():
    p = parsed()
    assert "productcatalogservice::EXTRA_LATENCY" in p.config_nodes
    # EXTRA_LATENCY must NOT have produced any call edge — it is the hidden surface.
    for src, dst in p.call_edges:
        assert "EXTRA_LATENCY" not in src and "EXTRA_LATENCY" not in dst


def test_file_ownership_by_directory():
    owners = parsed().file_owner
    assert owners["src/frontend/main.go"] == "frontend"
    assert owners["src/productcatalogservice/server.go"] == "productcatalogservice"


def test_productcatalog_endpoints_extracted():
    eps = parsed().endpoints["productcatalogservice"]
    assert set(eps) == {"ListProducts", "GetProduct", "SearchProducts"}


_RESOURCE_MANIFEST = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myservice
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: server
          env:
            - name: DISABLE_PROFILER
              value: "1"
          resources:
            limits:
              cpu: 200m
              memory: 128Mi
            requests:
              cpu: 100m
              memory: 64Mi
"""


def test_resource_and_replica_config_nodes(tmp_path):
    (tmp_path / "kubernetes-manifests.yaml").write_text(_RESOURCE_MANIFEST)
    cfg = set(parse_repo(tmp_path).config_nodes)

    # scale + resource knobs become inert config nodes
    assert "myservice::REPLICAS" in cfg
    assert "myservice::LIMITS_CPU" in cfg
    assert "myservice::REQUESTS_CPU" in cfg
    assert "myservice::LIMITS_MEMORY" in cfg
    assert "myservice::REQUESTS_MEMORY" in cfg
    # ordinary non-addr env var still becomes a config node alongside them
    assert "myservice::DISABLE_PROFILER" in cfg


def test_resource_knobs_robust_to_missing_fields(tmp_path):
    # No replicas, no resources: only the env-var config node should appear,
    # and no resource/replica knobs should be invented.
    manifest = (
        "apiVersion: apps/v1\n"
        "kind: Deployment\n"
        "metadata:\n"
        "  name: bareservice\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        "        - name: server\n"
        "          env:\n"
        "            - name: DISABLE_PROFILER\n"
        "              value: \"1\"\n"
    )
    (tmp_path / "kubernetes-manifests.yaml").write_text(manifest)
    cfg = parse_repo(tmp_path).config_nodes

    assert "bareservice::DISABLE_PROFILER" in cfg
    for knob in ("REPLICAS", "LIMITS_CPU", "REQUESTS_CPU", "LIMITS_MEMORY", "REQUESTS_MEMORY"):
        assert f"bareservice::{knob}" not in cfg
    # resource knobs create no edges
    assert parse_repo(tmp_path).call_edges == []
