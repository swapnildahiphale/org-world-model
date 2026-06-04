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
