import json
from scripts.run import run


def test_offline_run_produces_a_brier_number(tmp_path):
    out = tmp_path / "results.json"
    res = run(offline=True, out_path=out)
    assert isinstance(res["brier_overall"], float)
    assert "brier_hidden" in res and "brier_given" in res
    assert res["n_scenarios"] == 2
    assert out.exists()
    assert json.loads(out.read_text())["brier_overall"] == res["brier_overall"]
