import json
from owm.scenarios import Scenario, default_scenarios, emit, content_hash


def test_roundtrip_dict():
    s = Scenario(id="x", stratum="hidden", role="teach", hidden_edge="h",
                 touched_configs=["productcatalogservice::EXTRA_LATENCY"], s_t="healthy")
    assert Scenario.from_dict(s.to_dict()) == s


def test_default_set_has_given_and_hidden_family():
    scs = {s.id: s for s in default_scenarios()}
    strata = {s.stratum for s in scs.values()}
    assert strata == {"given", "hidden"}
    roles = {s.role for s in scs.values() if s.stratum == "hidden"}
    assert roles == {"teach", "transfer", "negative"}


def test_teach_and_transfer_share_hidden_edge_and_class_node():
    by_role = {s.role: s for s in default_scenarios() if s.stratum == "hidden"}
    teach, transfer, negative = by_role["teach"], by_role["transfer"], by_role["negative"]
    # same hidden edge + same class-node origin => genuine transfer, not a new edge
    assert teach.hidden_edge == transfer.hidden_edge
    assert teach.touched_configs == transfer.touched_configs
    # negative routes through a DIFFERENT class node and has no hidden edge
    assert negative.touched_configs != teach.touched_configs
    assert negative.hidden_edge is None


def test_emit_and_hash_are_deterministic(tmp_path):
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    emit(default_scenarios(), p1)
    emit(default_scenarios(), p2)
    assert content_hash(p1) == content_hash(p2)
    assert isinstance(json.loads(p1.read_text()), list)
