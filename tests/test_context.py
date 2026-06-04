from owm.scenarios import Scenario
from owm.context import context_pack, as_llm_text, facts


def test_pack_contains_topology_and_change(seed_cwm):
    sc = Scenario(id="t", stratum="given", touched_services=("productcatalogservice",))
    pack = context_pack(seed_cwm, sc)
    assert "frontend" in pack["services"]
    assert ("frontend", "productcatalogservice") in [tuple(e) for e in pack["call_edges"]]
    assert pack["change"]["touched_services"] == ["productcatalogservice"]


def test_llm_text_exposes_every_fact_in_pack(seed_cwm):
    sc = Scenario(id="t", stratum="hidden", touched_configs=("productcatalogservice::EXTRA_LATENCY",))
    incs = [{"origin": "productcatalogservice::EXTRA_LATENCY",
             "impacted": ["frontend", "recommendationservice"]}]
    pack = context_pack(seed_cwm, sc, incidents=incs)
    text = as_llm_text(pack, include_incidents=True)
    for svc in pack["services"]:
        assert svc in text
    for u, v in pack["call_edges"]:
        assert f"{u} -> {v}" in text
    assert "productcatalogservice::EXTRA_LATENCY" in text   # incident origin is exposed


def test_stateless_hides_incidents_rag_shows_them(seed_cwm):
    sc = Scenario(id="t", stratum="hidden", touched_configs=("productcatalogservice::EXTRA_LATENCY",))
    incs = [{"origin": "productcatalogservice::EXTRA_LATENCY", "impacted": ["frontend"]}]
    pack = context_pack(seed_cwm, sc, incidents=incs)
    assert "Incident history" not in as_llm_text(pack, include_incidents=False)
    assert "Incident history" in as_llm_text(pack, include_incidents=True)


def test_facts_are_stable(seed_cwm):
    sc = Scenario(id="t", stratum="given", touched_services=("productcatalogservice",))
    pack = context_pack(seed_cwm, sc)
    assert "frontend" in facts(pack)
    assert "frontend->productcatalogservice" in facts(pack)
