import pytest
from scripts.run import run_full
from owm.baselines import SELF_CONSISTENCY_N, StatelessLLM, RAGLLM


# --- spy helper ---

def _capture(cls, lst, *a, **kw):
    """Instantiate cls, record the instance in lst, return it."""
    obj = cls(*a, **kw)
    lst.append(obj)
    return obj


# --- TDD tests for self_consistency_n wiring ---

def test_run_full_default_self_consistency_n_is_gte_5(monkeypatch, tmp_path):
    stateless_instances = []
    rag_instances = []

    monkeypatch.setattr(
        "scripts.run.StatelessLLM",
        lambda *a, **kw: _capture(StatelessLLM, stateless_instances, *a, **kw),
    )
    monkeypatch.setattr(
        "scripts.run.RAGLLM",
        lambda *a, **kw: _capture(RAGLLM, rag_instances, *a, **kw),
    )

    run_full(offline=True, out_dir=tmp_path)

    assert len(stateless_instances) == 1
    assert len(rag_instances) == 1
    assert stateless_instances[0].n >= 5
    assert rag_instances[0].n >= 5
    assert stateless_instances[0].n == SELF_CONSISTENCY_N
    assert rag_instances[0].n == SELF_CONSISTENCY_N


def test_run_full_honors_self_consistency_n_override(monkeypatch, tmp_path):
    stateless_instances = []
    rag_instances = []

    monkeypatch.setattr(
        "scripts.run.StatelessLLM",
        lambda *a, **kw: _capture(StatelessLLM, stateless_instances, *a, **kw),
    )
    monkeypatch.setattr(
        "scripts.run.RAGLLM",
        lambda *a, **kw: _capture(RAGLLM, rag_instances, *a, **kw),
    )

    run_full(offline=True, out_dir=tmp_path, self_consistency_n=2)

    assert stateless_instances[0].n == 2
    assert rag_instances[0].n == 2


# --- existing test ---

def test_full_offline_pipeline_emits_all_predictors_and_deepening(tmp_path):
    res = run_full(offline=True, out_dir=tmp_path)
    # every predictor scored on the HIDDEN stratum
    for name in ("engine", "stateless_llm", "rag_llm", "bfs"):
        assert name in res["hidden_brier"]
        assert isinstance(res["hidden_brier"][name], float)
    # deepening present for engine and the learning-disabled ablation
    assert "engine" in res["deepening"] and "engine_no_learning" in res["deepening"]
    # artifacts written
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "reliability.png").exists()
    # the ablation does NOT deepen; the engine does
    d = res["deepening"]
    assert d["engine"]["brier_post"] <= d["engine"]["brier_pre"]
    assert d["engine_no_learning"]["brier_post"] == d["engine_no_learning"]["brier_pre"]
