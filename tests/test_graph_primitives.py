from owm.graph import CausalWorldModel, Kind, Rel, PROPAGATING_RELS, GIVEN, LEARNED


def test_config_kind_and_couples_rel_exist():
    assert Kind.CONFIG == "config"
    assert Rel.COUPLES == "couples"


def test_couples_is_not_a_plain_propagating_rel():
    # COUPLES flows cause->effect (config -> service), the OPPOSITE direction
    # of CALLS. propagate.py handles it explicitly; callers_of() must NOT pick
    # it up, or blast radius would traverse it backwards.
    assert Rel.COUPLES not in PROPAGATING_RELS


def test_config_node_can_be_added_and_is_isolated_by_default():
    cwm = CausalWorldModel()
    cwm.add_node("productcatalogservice::EXTRA_LATENCY", Kind.CONFIG)
    assert "productcatalogservice::EXTRA_LATENCY" in cwm.nodes_of(Kind.CONFIG)
    # No outgoing edges in the seed — this is what makes the coupling discoverable.
    assert cwm.g.out_degree("productcatalogservice::EXTRA_LATENCY") == 0


def test_couples_edge_records_learned_provenance():
    cwm = CausalWorldModel()
    cwm.add_node("cfg", Kind.CONFIG)
    cwm.add_node("frontend", Kind.SERVICE)
    cwm.add_edge("cfg", "frontend", Rel.COUPLES, weight=0.6, source=LEARNED)
    u, v, data = next(iter(cwm.edges_view(source=LEARNED)))
    assert (u, v) == ("cfg", "frontend")
    assert data["rel"] == Rel.COUPLES and data["source"] == LEARNED
