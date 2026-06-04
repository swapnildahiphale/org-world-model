from pathlib import Path
from owm.codegraph import parse_repo
from owm.topology import seed_graph, beta_weight
from owm.graph import Kind, Rel, GIVEN

FIXTURE = Path(__file__).parent / "fixtures" / "ob_mini"


def build():
    return seed_graph(parse_repo(FIXTURE))


def test_services_and_config_are_nodes():
    cwm = build()
    assert set(cwm.services) == {
        "frontend", "recommendationservice", "productcatalogservice", "currencyservice"
    }
    assert "productcatalogservice::EXTRA_LATENCY" in cwm.nodes_of(Kind.CONFIG)


def test_all_seed_edges_are_given():
    cwm = build()
    assert list(cwm.edges_view(source="learned")) == []
    assert len(list(cwm.edges_view(source=GIVEN))) > 0


def test_call_edges_have_naive_half_prior():
    cwm = build()
    # frontend --calls--> productcatalogservice, Beta(1,1) => weight 0.5
    data = cwm.g.get_edge_data("frontend", "productcatalogservice", key=Rel.CALLS)
    assert data["alpha"] == 1 and data["beta"] == 1
    assert abs(data["weight"] - 0.5) < 1e-9


def test_config_node_has_no_outgoing_edges():
    cwm = build()
    assert cwm.g.out_degree("productcatalogservice::EXTRA_LATENCY") == 0


def test_beta_weight_is_posterior_mean():
    assert abs(beta_weight(1, 1) - 0.5) < 1e-9
    assert abs(beta_weight(9, 1) - 0.9) < 1e-9
