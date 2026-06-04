from owm.graph import CausalWorldModel, Kind, Rel, LEARNED
from owm.topology import beta_variance
from owm.propagate import predict_blast
from owm.learn import observe, update_edge


def test_observe_impacted_moves_weight_up_and_shrinks_variance():
    cwm = CausalWorldModel()
    cwm.add_node("c", Kind.CONFIG)
    cwm.add_node("frontend", Kind.SERVICE, mitigation=0.0)
    cwm.add_edge("c", "frontend", Rel.COUPLES, weight=0.5, source=LEARNED, alpha=1, beta=1)
    before = beta_variance(1, 1)
    update_edge(cwm, "c", "frontend", Rel.COUPLES, impacted=True)
    d = cwm.g.get_edge_data("c", "frontend", key=Rel.COUPLES)
    assert d["alpha"] == 2 and d["beta"] == 1
    assert d["weight"] > 0.5
    assert beta_variance(d["alpha"], d["beta"]) < before


def test_surprise_adds_a_learned_coupling(seed_cwm):
    origin = "productcatalogservice::EXTRA_LATENCY"
    pre = predict_blast(seed_cwm, touched_configs=[origin], s_t="healthy")
    assert pre["frontend"] < 0.1  # engine gave near-zero -> the surprise condition
    observe(seed_cwm, origin, {"frontend", "recommendationservice"}, pre, learning_enabled=True)
    assert seed_cwm.g.has_edge(origin, "frontend", key=Rel.COUPLES)
    u, v, data = next(e for e in seed_cwm.edges_view(source=LEARNED) if e[1] == "frontend")
    assert data["saturation_sensitive"] is True


def test_learning_generalizes_to_transfer_not_negative(seed_cwm):
    cwm = seed_cwm
    teach = transfer = "productcatalogservice::EXTRA_LATENCY"   # same parsed knob (genuine transfer)
    negative = "productcatalogservice::DISABLE_PROFILER"              # different parsed knob on same service (locality)
    pre = predict_blast(cwm, touched_configs=[teach], s_t="healthy")
    observe(cwm, teach, {"frontend", "recommendationservice"}, pre, learning_enabled=True)

    post_transfer = predict_blast(cwm, touched_configs=[transfer], s_t="healthy")
    assert post_transfer["frontend"] > 0.1            # generalization, not memorization
    assert post_transfer["recommendationservice"] > 0.1

    post_negative = predict_blast(cwm, touched_configs=[negative], s_t="healthy")
    assert all(p < 1e-9 for p in post_negative.values())   # specificity: no over-fire


def test_learned_origin_is_a_genuinely_parsed_config_node(seed_cwm):
    """Honesty (Rule 2): the taught coupling attaches to a node the static parser
    actually produced — not a synthetic id invented by the harness."""
    from owm.scenarios import LATENCY_KNOB, BENIGN_KNOB
    from owm.graph import Kind
    parsed = seed_cwm.nodes_of(Kind.CONFIG)
    assert LATENCY_KNOB in parsed
    assert BENIGN_KNOB in parsed


def test_learning_disabled_is_a_noop(seed_cwm):
    before = seed_cwm.g.number_of_edges()
    observe(seed_cwm, "productcatalogservice::EXTRA_LATENCY", {"frontend"},
            {"frontend": 0.0}, learning_enabled=False)
    assert seed_cwm.g.number_of_edges() == before


# --- Week-2: optional CALLS-edge learning -----------------------------------

def _call_edge(cwm, caller, callee):
    return cwm.g.get_edge_data(caller, callee, key=Rel.CALLS)


def test_default_observe_leaves_call_edges_untouched():
    """Backward-compat: WITHOUT learn_call_edges, seeded CALLS edges are byte-for-byte
    unchanged — same Beta(1,1) seed (alpha=1, beta=1, weight=0.5) the topology set."""
    cwm = CausalWorldModel()
    cwm.add_node("frontend", Kind.SERVICE, mitigation=0.0)
    cwm.add_node("productcatalogservice", Kind.SERVICE, mitigation=0.0)
    cwm.add_edge("frontend", "productcatalogservice", Rel.CALLS,
                 weight=0.5, source="given", alpha=1.0, beta=1.0)

    # callee impacted, caller impacted -> would update IF the flag were on; it isn't.
    observe(cwm, "frontend::SOME_KNOB", {"frontend", "productcatalogservice"},
            {}, learn_call_edges=False)

    d = _call_edge(cwm, "frontend", "productcatalogservice")
    assert d["alpha"] == 1.0 and d["beta"] == 1.0
    assert d["weight"] == 0.5


def test_default_on_seed_fixture_leaves_all_call_edges_untouched(seed_cwm):
    """Same guarantee on the real seeded fixture: an incident that hits real callees
    must not perturb any CALLS edge when learning call edges is off (the default)."""
    before = {
        (u, v): (d["alpha"], d["beta"], d["weight"])
        for u, v, k, d in seed_cwm.g.edges(keys=True, data=True) if k == Rel.CALLS
    }
    observe(seed_cwm, "productcatalogservice::EXTRA_LATENCY",
            {"frontend", "productcatalogservice", "recommendationservice"},
            {}, learning_enabled=True)  # learn_call_edges defaults False
    after = {
        (u, v): (d["alpha"], d["beta"], d["weight"])
        for u, v, k, d in seed_cwm.g.edges(keys=True, data=True) if k == Rel.CALLS
    }
    assert after == before


def test_call_edge_learning_caller_and_callee_impacted_raises_weight():
    """callee in blast AND caller in blast -> positive obs: alpha++, weight rises above seed."""
    cwm = CausalWorldModel()
    cwm.add_node("frontend", Kind.SERVICE, mitigation=0.0)
    cwm.add_node("productcatalogservice", Kind.SERVICE, mitigation=0.0)
    cwm.add_edge("frontend", "productcatalogservice", Rel.CALLS,
                 weight=0.5, source="given", alpha=1.0, beta=1.0)

    observe(cwm, "frontend::SOME_KNOB", {"frontend", "productcatalogservice"},
            {}, learn_call_edges=True)

    d = _call_edge(cwm, "frontend", "productcatalogservice")
    assert d["alpha"] == 2.0 and d["beta"] == 1.0
    assert d["weight"] > 0.5


def test_call_edge_learning_callee_impacted_caller_not_drops_weight():
    """callee in blast but caller NOT -> failure didn't propagate: beta++, weight drops below seed."""
    cwm = CausalWorldModel()
    cwm.add_node("frontend", Kind.SERVICE, mitigation=0.0)
    cwm.add_node("productcatalogservice", Kind.SERVICE, mitigation=0.0)
    cwm.add_edge("frontend", "productcatalogservice", Rel.CALLS,
                 weight=0.5, source="given", alpha=1.0, beta=1.0)

    observe(cwm, "productcatalogservice::SOME_KNOB", {"productcatalogservice"},
            {}, learn_call_edges=True)

    d = _call_edge(cwm, "frontend", "productcatalogservice")
    assert d["alpha"] == 1.0 and d["beta"] == 2.0
    assert d["weight"] < 0.5


def test_call_edge_learning_callee_not_impacted_is_untouched():
    """No signal when the callee isn't in the blast: edge stays at its Beta(1,1) seed."""
    cwm = CausalWorldModel()
    cwm.add_node("frontend", Kind.SERVICE, mitigation=0.0)
    cwm.add_node("productcatalogservice", Kind.SERVICE, mitigation=0.0)
    cwm.add_edge("frontend", "productcatalogservice", Rel.CALLS,
                 weight=0.5, source="given", alpha=1.0, beta=1.0)

    # Only the caller is impacted; callee is clean -> this edge carries no signal.
    observe(cwm, "frontend::SOME_KNOB", {"frontend"}, {}, learn_call_edges=True)

    d = _call_edge(cwm, "frontend", "productcatalogservice")
    assert d["alpha"] == 1.0 and d["beta"] == 1.0
    assert d["weight"] == 0.5
