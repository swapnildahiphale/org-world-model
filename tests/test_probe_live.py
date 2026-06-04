"""Unit tests for the live trace probe (NO network / NO cluster calls).

Primary transport is Tempo's TraceQL metrics API; a FakeTempoClient returns
canned envelopes in the EXACT live shape (incl. the extra `p` label on quantile
queries and a multi-service+status rate case). The Prom/VM fallback is also
covered. We assert parsing, the degraded-set decision rule, and round-trip a
written run-file through groundtruth.journey_probe for pipeline compatibility.
"""
from __future__ import annotations

import json

from groundtruth import probe_live
from groundtruth.probe_live import (
    ERR_THRESH,
    LAT_MULT,
    PROM_STRATEGY,
    capture_runs,
    degraded_services,
    prom_error_rate_query,
    prom_p95_latency_query,
    tempo_error_rate_numerator_query,
    tempo_p95_latency_query,
    tempo_total_rate_query,
    write_run_file,
)
from groundtruth.journey_probe import aggregate_runs, probe_file_to_ground_truth


def _svc_label(name: str) -> dict:
    return {"key": probe_live.SERVICE_LABEL, "value": {"stringValue": name}}


def _p_label() -> dict:
    # The extra label Tempo adds on quantile_over_time queries; must be ignored.
    return {"key": "p", "value": {"doubleValue": 0.95}}


class FakeTempoClient:
    """Returns canned Tempo envelopes keyed on the TraceQL query family.

    Distinguishes the three query builders by substring so it stays correct even
    if the constants change:
      * quantile_over_time  -> p95 envelope (each series carries an extra `p` label)
      * {status=error} ...  -> errored-span rate envelope
      * {} | rate()         -> total-span rate envelope
    Each canned input is {service: value}; we wrap it in the exact live envelope
    and parse it back through the production parser so real parsing is exercised.
    """

    def __init__(self, p95, error_rate_num, total_rate):
        self._p95 = p95
        self._err = error_rate_num
        self._total = total_rate
        self.queries: list[str] = []

    def query(self, traceql: str, *, since=None, start=None, end=None) -> dict[str, float]:
        self.queries.append(traceql)
        if "quantile_over_time" in traceql:
            return self._envelope(self._p95, with_p_label=True)
        if f"status={probe_live.ERROR_STATUS}" in traceql:
            return self._envelope(self._err)
        if "rate()" in traceql:
            return self._envelope(self._total)
        raise AssertionError(f"unexpected TraceQL: {traceql}")

    @staticmethod
    def _envelope(values: dict, *, with_p_label: bool = False) -> dict[str, float]:
        series = []
        for svc, val in values.items():
            labels = [_svc_label(svc)]
            if with_p_label:
                labels.append(_p_label())
            series.append({"labels": labels, "value": val})
        payload = {"series": series, "metrics": {"inspectedSpans": 0}}
        return probe_live.TempoMetricsClient._parse_series(payload)


# Shared baseline: every service had a healthy 100ms p95 before injection.
BASELINE = {
    "frontend": 0.100,
    "recommendationservice": 0.100,
    "currencyservice": 0.100,
    "cartservice": 0.100,
}


# --- Protocol / parsing -------------------------------------------------------

def test_fake_tempo_satisfies_metricsclient_protocol():
    fake = FakeTempoClient({}, {}, {})
    assert isinstance(fake, probe_live.MetricsClient)


def test_tempo_parser_maps_service_to_value_and_ignores_p_label():
    # EXACT real capture shape, with the extra `p` label on a quantile series.
    payload = {
        "series": [
            {
                "labels": [
                    {"key": "resource.service.name",
                     "value": {"stringValue": "agentgateway-proxy"}},
                    {"key": "p", "value": {"doubleValue": 0.95}},
                ],
                "value": 0.03076954981143621,
            },
        ],
        "metrics": {"inspectedTraces": 1, "inspectedSpans": 3},
    }
    parsed = probe_live.TempoMetricsClient._parse_series(payload)
    assert parsed == {"agentgateway-proxy": 0.03076954981143621}  # `p` ignored


def test_tempo_parser_multi_service_and_status_and_robust_to_missing_labels():
    # A rate-by-(service,status) result: multiple services, a status label, and
    # one malformed series with no service label (must be skipped, not crash).
    payload = {
        "series": [
            {"labels": [_svc_label("frontend"),
                        {"key": "status", "value": {"stringValue": "error"}}],
             "value": 0.5},
            {"labels": [_svc_label("cartservice"),
                        {"key": "status", "value": {"stringValue": "ok"}}],
             "value": 9.0},
            {"labels": [{"key": "status", "value": {"stringValue": "ok"}}],  # no service
             "value": 1.0},
            {"labels": [_svc_label("emailservice")], "value": float("nan")},  # NaN dropped
        ],
        "metrics": {},
    }
    parsed = probe_live.TempoMetricsClient._parse_series(payload)
    assert parsed == {"frontend": 0.5, "cartservice": 9.0}


def test_tempo_label_value_extracts_string_double_int():
    lv = probe_live.TempoMetricsClient._label_value
    assert lv({"stringValue": "x"}) == "x"
    assert lv({"doubleValue": 0.95}) == 0.95
    assert lv({"intValue": 3}) == 3
    assert lv({}) is None
    assert lv("notadict") is None


# --- Decision rule (Tempo strategy, the default) ------------------------------

def test_degraded_flags_latency_breach_error_breach_but_not_healthy_tempo():
    # frontend: p95 blown past LAT_MULT*baseline (latency breach), no errors.
    # recommendationservice: p95 fine, but error rate above ERR_THRESH (error breach).
    # currencyservice: p95 fine AND error rate below threshold -> healthy.
    # cartservice: p95 exactly AT LAT_MULT*baseline (strict > -> not degraded),
    #              error rate exactly AT threshold -> not degraded.
    p95 = {
        "frontend": LAT_MULT * BASELINE["frontend"] + 0.050,
        "recommendationservice": 0.110,
        "currencyservice": 0.090,
        "cartservice": LAT_MULT * BASELINE["cartservice"],
    }
    total = {"frontend": 100.0, "recommendationservice": 100.0,
             "currencyservice": 100.0, "cartservice": 100.0}
    err_num = {
        "frontend": 0.0,                       # 0%
        "recommendationservice": 5.0,          # 5% > 1% -> breach
        "currencyservice": 0.1,                # 0.1% < 1%
        "cartservice": ERR_THRESH * 100.0,     # exactly 1% -> not a breach
    }
    fake = FakeTempoClient(p95, err_num, total)

    degraded = degraded_services(fake, BASELINE, window="1m")  # default = TEMPO_STRATEGY

    assert "frontend" in degraded                 # latency breach
    assert "recommendationservice" in degraded    # error-rate breach
    assert "currencyservice" not in degraded       # under both thresholds
    assert "cartservice" not in degraded           # both exactly at boundary
    assert degraded == {"frontend", "recommendationservice"}


def test_tempo_error_rate_guards_divide_by_zero():
    # A service with zero total spans this window must yield error_rate 0.0,
    # not a ZeroDivisionError, and must not be flagged.
    p95 = {"idleservice": 0.05}                      # fine on latency
    total = {"idleservice": 0.0}                     # no traffic
    err_num = {"idleservice": 0.0}
    fake = FakeTempoClient(p95, err_num, total)
    assert degraded_services(fake, BASELINE, window="1m") == set()


def test_latency_not_evaluated_without_baseline_but_errors_still_are_tempo():
    # No baseline entry -> not latency-flagged even if huge; error breach still counts.
    p95 = {"newservice": 99.0}
    total = {"newservice": 10.0}
    err_num = {"newservice": 5.0}                    # 50% errors -> breach
    fake = FakeTempoClient(p95, err_num, total)
    assert degraded_services(fake, baseline_p95={}, window="1m") == {"newservice"}

    # Same huge latency, no errors, still no baseline -> not degraded.
    fake2 = FakeTempoClient({"newservice": 99.0}, {"newservice": 0.0}, {"newservice": 10.0})
    assert degraded_services(fake2, baseline_p95={}, window="1m") == set()


def test_capture_runs_returns_n_sorted_lists_tempo():
    p95 = {"frontend": 1.0, "recommendationservice": 1.0}  # both 10x baseline -> breach
    total = {"frontend": 50.0, "recommendationservice": 50.0}
    err_num = {"frontend": 0.0, "recommendationservice": 0.0}
    fake = FakeTempoClient(p95, err_num, total)

    runs = capture_runs(fake, BASELINE, window="30s", n_windows=4)

    assert len(runs) == 4
    for run in runs:
        assert run == sorted(run)
        assert run == ["frontend", "recommendationservice"]


# --- TraceQL query builders ---------------------------------------------------

def test_tempo_query_builders_reference_constants():
    p95q = tempo_p95_latency_query()
    errq = tempo_error_rate_numerator_query()
    totq = tempo_total_rate_query()
    assert p95q == ("{} | quantile_over_time(duration, 0.95) "
                    "by (resource.service.name)")
    assert errq == "{status=error} | rate() by (resource.service.name)"
    assert totq == "{} | rate() by (resource.service.name)"
    assert probe_live.SERVICE_LABEL in p95q
    assert probe_live.DURATION_INTRINSIC in p95q
    assert probe_live.ERROR_STATUS in errq


# --- Prom fallback (kept) -----------------------------------------------------

class FakePromClient:
    def __init__(self, p95, error_rate):
        self._p95 = p95
        self._error_rate = error_rate

    def query(self, promql: str) -> dict[str, float]:
        if "histogram_quantile" in promql:
            return dict(self._p95)
        if probe_live.PROM_SPAN_CALLS_TOTAL in promql:
            return dict(self._error_rate)
        raise AssertionError(f"unexpected query: {promql}")


def test_prom_fallback_strategy_still_works():
    p95 = {"frontend": LAT_MULT * 0.1 + 0.05, "currencyservice": 0.09}
    err = {"frontend": 0.0, "currencyservice": 0.0}
    fake = FakePromClient(p95, err)
    degraded = degraded_services(fake, {"frontend": 0.1, "currencyservice": 0.1},
                                 window="1m", strategy=PROM_STRATEGY)
    assert degraded == {"frontend"}


def test_prom_query_builders_reference_constants():
    lat = prom_p95_latency_query("1m")
    err = prom_error_rate_query("5m")
    assert "histogram_quantile(0.95" in lat
    assert probe_live.PROM_SPAN_DURATION_BUCKET in lat
    assert "[1m]" in lat
    assert probe_live.PROM_SPAN_CALLS_TOTAL in err
    assert probe_live.PROM_ERROR_STATUS_VALUE in err
    assert "[5m]" in err


def test_prom_client_parses_vector_envelope():
    payload = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {probe_live.PROM_SERVICE_LABEL: "frontend"}, "value": [1.0, "0.42"]},
                {"metric": {probe_live.PROM_SERVICE_LABEL: "cartservice"}, "value": [1.0, "NaN"]},
                {"metric": {"other": "x"}, "value": [1.0, "9.9"]},  # no service label
            ],
        },
    }
    parsed = probe_live.PromMetricsClient._parse_vector(payload)
    assert parsed == {"frontend": 0.42}


# --- Round-trip through the pipeline ------------------------------------------

def test_round_trip_write_then_ground_truth_two_thirds_majority(tmp_path):
    # frontend 3/3 (kept), recommendationservice 2/3 (kept), currencyservice 1/3 (dropped).
    runs = [
        ["frontend", "recommendationservice"],
        ["frontend", "recommendationservice"],
        ["currencyservice", "frontend"],
    ]
    out = tmp_path / "live-extra-latency.json"
    written = write_run_file("live-extra-latency", "hidden", runs, out)

    rec = json.loads(written.read_text())
    assert set(rec.keys()) == {"scenario_id", "stratum", "runs"}
    assert rec["scenario_id"] == "live-extra-latency"
    assert rec["stratum"] == "hidden"
    assert rec["runs"] == runs

    gt = probe_file_to_ground_truth(written)
    assert gt.scenario_id == "live-extra-latency"
    assert gt.stratum == "hidden"
    expected = aggregate_runs(runs)
    assert expected == {"frontend", "recommendationservice"}
    assert gt.impacted == expected


def test_round_trip_end_to_end_from_fake_tempo_client(tmp_path):
    # Drive the whole probe from the fake Tempo client, then round-trip.
    # frontend breaches latency every window; recommendationservice breaches errors.
    p95 = {
        "frontend": 1.0,                     # 10x baseline -> latency breach
        "recommendationservice": 0.110,      # fine on latency
        "currencyservice": 0.090,            # fine
    }
    total = {"frontend": 100.0, "recommendationservice": 100.0, "currencyservice": 100.0}
    err_num = {
        "frontend": 0.0,
        "recommendationservice": 20.0,       # 20% -> error breach
        "currencyservice": 0.0,
    }
    fake = FakeTempoClient(p95, err_num, total)
    runs = capture_runs(fake, BASELINE, window="1m", n_windows=3)

    out = tmp_path / "run.json"
    write_run_file("sc-fake", "given", runs, out)
    gt = probe_file_to_ground_truth(out)

    assert gt.scenario_id == "sc-fake"
    assert gt.stratum == "given"
    assert gt.impacted == {"frontend", "recommendationservice"}
