from owm.graph import CausalWorldModel, Kind, Rel, GIVEN, LEARNED
from owm.propagate import predict_blast


def _two_path_graph():
    """frontend --calls--> productcatalog (direct)
       frontend --calls--> recommendation --calls--> productcatalog
       Two paths from productcatalog's failure to frontend."""
    cwm = CausalWorldModel()
    for s in ("frontend", "recommendationservice", "productcatalogservice"):
        cwm.add_node(s, Kind.SERVICE, mitigation=0.0)
    cwm.add_edge("frontend", "productcatalogservice", Rel.CALLS, weight=0.5, source=GIVEN)
    cwm.add_edge("frontend", "recommendationservice", Rel.CALLS, weight=0.5, source=GIVEN)
    cwm.add_edge("recommendationservice", "productcatalogservice", Rel.CALLS, weight=0.5, source=GIVEN)
    return cwm


def test_noisy_or_composes_two_paths():
    cwm = _two_path_graph()
    blast = predict_blast(cwm, touched_services=["productcatalogservice"], s_t="healthy")
    # recommendation: single parent pc(1.0)*0.5 = 0.5
    assert abs(blast["recommendationservice"] - 0.5) < 1e-6
    # frontend: noisy-OR of direct (1.0*0.5) and via-rec (0.5*0.5):
    #   1 - (1-0.5)*(1-0.25) = 0.625
    assert abs(blast["frontend"] - 0.625) < 1e-6


def test_config_change_on_seed_predicts_near_zero():
    """Pre-learning: a config node has no COUPLES edges, so nothing is impacted.
    This near-zero prediction is exactly the 'surprise' condition for learning."""
    cwm = _two_path_graph()
    cwm.add_node("productcatalogservice::EXTRA_LATENCY", Kind.CONFIG)
    blast = predict_blast(cwm, touched_configs=["productcatalogservice::EXTRA_LATENCY"], s_t="healthy")
    assert all(p < 1e-9 for p in blast.values())


def test_mitigation_lowers_impact():
    cwm = _two_path_graph()
    base = predict_blast(cwm, touched_services=["productcatalogservice"], s_t="healthy")["frontend"]
    cwm.g.nodes["frontend"]["mitigation"] = 0.5
    mit = predict_blast(cwm, touched_services=["productcatalogservice"], s_t="healthy")["frontend"]
    assert mit < base


def test_state_amplifies_saturation_sensitive_coupling():
    cwm = _two_path_graph()
    cwm.add_node("cfg", Kind.CONFIG)
    cwm.add_edge("cfg", "frontend", Rel.COUPLES, weight=0.5, source=LEARNED,
                 saturation_sensitive=True)
    healthy = predict_blast(cwm, touched_configs=["cfg"], s_t="healthy")["frontend"]
    saturated = predict_blast(cwm, touched_configs=["cfg"], s_t="saturated")["frontend"]
    assert saturated > healthy
