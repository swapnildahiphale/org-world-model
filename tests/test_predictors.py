from owm.scenarios import Scenario
from owm.context import context_pack
from owm.engine import OWMEngine
from owm.baselines import StatelessLLM, BFSBaseline, FakeClient, parse_service_set


def test_bfs_reaches_callers_for_code_change(seed_cwm):
    sc = Scenario(id="g", stratum="given", touched_services=("productcatalogservice",))
    pred = BFSBaseline(seed_cwm).predict(sc)
    assert pred.probs["frontend"] == 1.0
    assert pred.probs["recommendationservice"] == 1.0


def test_bfs_predicts_zero_for_config_change(seed_cwm):
    sc = Scenario(id="h", stratum="hidden", touched_configs=("productcatalogservice::EXTRA_LATENCY",))
    pred = BFSBaseline(seed_cwm).predict(sc)
    assert all(p == 0.0 for p in pred.probs.values())


def test_parse_service_set_filters_unknown():
    out = parse_service_set('noise ["frontend","nope"] tail', ["frontend", "cartservice"])
    assert out == {"frontend"}


def test_llm_confidence_is_self_consistency_frequency(seed_cwm):
    sc = Scenario(id="h", stratum="hidden", touched_configs=("productcatalogservice::EXTRA_LATENCY",))
    pack = context_pack(seed_cwm, sc)
    # 5 samples: frontend in 4/5, recommendationservice in 2/5
    client = FakeClient([
        "[]",
        '["frontend","recommendationservice"]',
        '["frontend"]',
        '["frontend","recommendationservice"]',
        '["frontend"]',
    ])
    pred = StatelessLLM(client, n=5).predict(sc, pack)
    assert abs(pred.probs["frontend"] - 0.8) < 1e-9
    assert abs(pred.probs["recommendationservice"] - 0.4) < 1e-9


def test_engine_learning_disabled_is_a_noop(seed_cwm):
    sc = Scenario(id="h", stratum="hidden", touched_configs=("productcatalogservice::EXTRA_LATENCY",))
    before = seed_cwm.g.number_of_edges()
    OWMEngine(seed_cwm, learning_enabled=False).learn(sc, {"frontend", "recommendationservice"})
    assert seed_cwm.g.number_of_edges() == before


def test_engine_learns_then_generalizes_to_transfer(seed_cwm):
    eng = OWMEngine(seed_cwm, learning_enabled=True)
    teach = Scenario(id="teach", stratum="hidden", role="teach",
                     touched_configs=("productcatalogservice::EXTRA_LATENCY",))
    eng.learn(teach, {"frontend", "recommendationservice"})
    transfer = Scenario(id="transfer", stratum="hidden", role="transfer",
                        touched_configs=("productcatalogservice::EXTRA_LATENCY",))
    assert eng.predict(transfer).probs["frontend"] > 0.1
