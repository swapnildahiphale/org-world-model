"""Automated trace probe: measure the degraded-service set per fault injection
from Tempo's per-service RED metrics and emit the run-list JSON the existing
pipeline already consumes.

Instead of a synthetic browse/cart/checkout journey probe, this reads the
per-service RED metrics Tempo derives from real traces (span duration + span
status) and decides, per measurement window, which services breached their SLO.
Each window becomes one inner list of the `runs` array; the downstream pipeline
(`groundtruth.journey_probe.probe_file_to_ground_truth`) folds the windows into
one GroundTruth via 2/3-majority.

Output record shape (EXACTLY what the pipeline consumes):
    {"scenario_id": str, "stratum": str, "runs": [[svc, ...], [svc, ...], ...]}

-------------------------------------------------------------------------------
TWO TRANSPORTS, TEMPO IS PRIMARY.

On THIS cluster the Tempo metrics-generator does NOT remote-write to
VictoriaMetrics, so `traces_spanmetrics_*` exists nowhere in Prometheus/VM.
Per-service RED lives ONLY in Tempo's TraceQL metrics API. `TempoMetricsClient`
(GET {base_url}/api/metrics/query?q=<TraceQL>) is therefore the PRIMARY
transport and the CLI default.

`PromMetricsClient` (Prometheus/VM `/api/v1/query` over `traces_spanmetrics_*`)
is kept as a documented FALLBACK for clusters that DO remote-write the
span-metrics. Select it with `--transport prom`.

VERIFIED against the live cluster (Tempo v2.9.0):
  - endpoint: GET {base_url}/api/metrics/query?q=<urlencoded TraceQL>
  - time window: optional `since=1h` OR `start=<unix_secs>&end=<unix_secs>`
  - service label key: resource.service.name
  - status intrinsic: status (values unset/ok/error); duration intrinsic: duration (s)
  - response envelope: {"series":[{"labels":[{"key":..,"value":{"stringValue"|
    "doubleValue"|"intValue":..}}, ...], "value": <float>}], "metrics":{...}}
    quantile queries add an extra `p` label (doubleValue) -- ignored.

The Prom-side metric/label constants remain PLACEHOLDERS for fallback clusters;
verify them against that cluster's metrics-generator before trusting a Prom run.
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

# --- SLO thresholds (the degraded-set decision rule; spec §7) -----------------
LAT_MULT = 2.0    # degraded if windowed p95 > LAT_MULT * baseline p95
ERR_THRESH = 0.01  # degraded if windowed error rate > 1%

# --- TraceQL metric / label names (PRIMARY; verified live, Tempo v2.9.0) ------
SERVICE_LABEL = "resource.service.name"   # per-service grouping label
DURATION_INTRINSIC = "duration"           # span duration intrinsic (seconds)
ERROR_STATUS = "error"                     # status intrinsic value marking an error
TEMPO_QUERY_PATH = "/api/metrics/query"   # Tempo TraceQL metrics endpoint path

# --- PromQL metric / label names (FALLBACK; PLACEHOLDERS) ---------------------
# Only used with --transport prom on clusters that remote-write the Tempo
# span-metrics. Verify these against that cluster's metrics-generator: the
# duration metric may be `traces_spanmetrics_duration_seconds_bucket` /
# `duration_bucket`; the service label may be `service` or `service_name`;
# status may surface as `status_code="STATUS_CODE_ERROR"` or an http range.
PROM_SPAN_DURATION_BUCKET = "traces_spanmetrics_latency_bucket"  # histogram bucket metric
PROM_SPAN_CALLS_TOTAL = "traces_spanmetrics_calls_total"          # request/span counter
PROM_SERVICE_LABEL = "service"                                    # per-service label key
PROM_STATUS_LABEL = "status_code"                                 # span status label key
PROM_ERROR_STATUS_VALUE = "STATUS_CODE_ERROR"                     # value marking an errored span


@runtime_checkable
class MetricsClient(Protocol):
    """Transport seam so tests can inject a fake (no network in tests).

    `query` runs one instant metrics query (TraceQL for Tempo, PromQL for
    Prometheus/VM) and returns the result collapsed to {service-label value:
    scalar value}. Series missing the service label or carrying a non-finite
    value are omitted by the implementation. Implementations may accept
    transport-specific keyword args (e.g. Tempo's `since`/`start`/`end`); the
    decision logic calls `query` positionally so any client type is usable.
    """

    def query(self, query: str) -> dict[str, float]:
        ...


# --- TraceQL query builders (PRIMARY) -----------------------------------------
# Each references the module-top TraceQL constants so a single edit there
# repoints every query at the verified live names.

def tempo_p95_latency_query() -> str:
    """Per-service p95 span duration via TraceQL metrics.

    {} | quantile_over_time(duration, 0.95) by (resource.service.name)

    Returns one series per service (plus an extra `p` label the parser ignores);
    the series `value` is the p95 duration in seconds. The measurement window is
    NOT part of the query string -- it is passed to the client as
    `since`/`start`/`end` against the Tempo endpoint.
    """
    return (
        f"{{}} | quantile_over_time({DURATION_INTRINSIC}, 0.95) "
        f"by ({SERVICE_LABEL})"
    )


def tempo_error_rate_numerator_query() -> str:
    """Per-service errored-span rate: {status=error} | rate() by (service)."""
    return f"{{status={ERROR_STATUS}}} | rate() by ({SERVICE_LABEL})"


def tempo_total_rate_query() -> str:
    """Per-service total span rate (denominator): {} | rate() by (service)."""
    return f"{{}} | rate() by ({SERVICE_LABEL})"


# --- PromQL query builders (FALLBACK) -----------------------------------------

def prom_p95_latency_query(window: str) -> str:
    """Per-service p95 span duration over `window` (e.g. "1m"), Prom/VM.

    histogram_quantile(0.95, sum by (service, le)(rate(<bucket>[<window>])))
    """
    return (
        f"histogram_quantile(0.95, sum by ({PROM_SERVICE_LABEL}, le)"
        f"(rate({PROM_SPAN_DURATION_BUCKET}[{window}])))"
    )


def prom_error_rate_query(window: str) -> str:
    """Per-service error rate = errored-span rate / total-span rate, Prom/VM.

    A bare `/` would drop services with no errors (no matching numerator series),
    so we OR in an explicit 0 for every service present in the denominator,
    keeping a healthy service in the result with error_rate == 0.0.
    """
    numer = (
        f"sum by ({PROM_SERVICE_LABEL})"
        f'(rate({PROM_SPAN_CALLS_TOTAL}{{{PROM_STATUS_LABEL}="{PROM_ERROR_STATUS_VALUE}"}}[{window}]))'
    )
    denom = f"sum by ({PROM_SERVICE_LABEL})(rate({PROM_SPAN_CALLS_TOTAL}[{window}]))"
    return f"({numer} / {denom}) or ({denom} * 0)"


# --- Concrete transports ------------------------------------------------------

class TempoMetricsClient:
    """PRIMARY transport: Tempo TraceQL metrics API (GET /api/metrics/query).

    Stdlib only (urllib + json), no new deps. Parses the OTLP-ish envelope::

        {"series":[{"labels":[{"key":"resource.service.name",
                               "value":{"stringValue":"frontend"}},
                              {"key":"p","value":{"doubleValue":0.95}}],
                    "value": 0.0307...}],
         "metrics":{...}}

    into {resource.service.name value: float(series.value)}. The extra `p` label
    on quantile queries is ignored; series missing the service label or with a
    non-finite value are skipped. Robust to missing/extra label entries.
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0):
        # base_url is e.g. "http://localhost:3200" or the in-cluster Tempo svc.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(self, traceql: str, *, since: str | None = None,
              start: int | None = None, end: int | None = None) -> dict[str, float]:
        params: list[tuple[str, str]] = [("q", traceql)]
        # Time window: prefer explicit start/end; else a relative `since`.
        if start is not None and end is not None:
            params.append(("start", str(start)))
            params.append(("end", str(end)))
        elif since is not None:
            params.append(("since", since))
        url = f"{self.base_url}{TEMPO_QUERY_PATH}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (trusted internal endpoint)
            payload = json.loads(resp.read().decode("utf-8"))
        return self._parse_series(payload)

    @staticmethod
    def _label_value(value_obj: dict):
        """Extract the scalar from a TraceQL label value object.

        value objects look like {"stringValue": "x"} / {"doubleValue": 0.95} /
        {"intValue": 3}. Returns the first present, else None.
        """
        if not isinstance(value_obj, dict):
            return None
        for key in ("stringValue", "doubleValue", "intValue"):
            if key in value_obj:
                return value_obj[key]
        return None

    @classmethod
    def _parse_series(cls, payload: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        for series in payload.get("series", []) or []:
            svc = None
            for label in series.get("labels", []) or []:
                if label.get("key") == SERVICE_LABEL:
                    svc = cls._label_value(label.get("value", {}))
                    break  # ignore the extra `p` label and any others
            if svc is None:
                continue
            raw = series.get("value")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val != val:  # NaN
                continue
            out[str(svc)] = val
        return out


class PromMetricsClient:
    """FALLBACK transport: Prometheus / VictoriaMetrics `/api/v1/query`.

    For clusters that DO remote-write the Tempo span-metrics. Stdlib only.
    Parses the standard instant-query vector envelope::

        {"status":"success",
         "data":{"resultType":"vector",
                 "result":[{"metric":{"service":"frontend",...},
                            "value":[<ts>, "<stringified float>"]}, ...]}}

    into {metric[PROM_SERVICE_LABEL]: float(value)}. Series missing the service
    label or carrying a non-finite value are skipped.
    """

    def __init__(self, endpoint: str, *, timeout: float = 30.0):
        # endpoint is the base URL, e.g. "http://victoria-metrics.monitoring:8428"
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def query(self, promql: str) -> dict[str, float]:
        url = f"{self.endpoint}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (trusted internal endpoint)
            payload = json.loads(resp.read().decode("utf-8"))
        return self._parse_vector(payload)

    @staticmethod
    def _parse_vector(payload: dict) -> dict[str, float]:
        if payload.get("status") != "success":
            raise RuntimeError(f"PromQL query failed: {payload.get('error', payload)}")
        data = payload.get("data", {})
        if data.get("resultType") != "vector":
            raise RuntimeError(f"expected vector result, got {data.get('resultType')!r}")
        out: dict[str, float] = {}
        for series in data.get("result", []):
            svc = series.get("metric", {}).get(PROM_SERVICE_LABEL)
            if svc is None:
                continue
            raw = series.get("value", [None, None])[1]
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val != val:  # NaN
                continue
            out[svc] = val
        return out


# --- Query strategies ---------------------------------------------------------
# A strategy bundles the transport-appropriate p95 + error-rate query builders,
# so the decision logic stays transport-agnostic. `window` is a string ("1m");
# for Tempo it is the relative lookback passed to the client as `since=<window>`
# (NOT part of the TraceQL string); for Prom it is interpolated into the range
# selector. The Tempo functions pass it as a kwarg, which the Tempo client and
# the test fake accept and the Prom client never sees.

def tempo_p95(client: MetricsClient, window: str) -> dict[str, float]:
    return client.query(tempo_p95_latency_query(), since=window)


def tempo_error_rate(client: MetricsClient, window: str) -> dict[str, float]:
    """error_rate = rate(status=error) / rate(all), per service; 0.0 if no traffic."""
    errors = client.query(tempo_error_rate_numerator_query(), since=window)
    totals = client.query(tempo_total_rate_query(), since=window)
    rates: dict[str, float] = {}
    for svc, total in totals.items():
        # Guard divide-by-zero: a service with no spans this window has rate 0.0.
        rates[svc] = (errors.get(svc, 0.0) / total) if total > 0 else 0.0
    return rates


def prom_p95(client: MetricsClient, window: str) -> dict[str, float]:
    return client.query(prom_p95_latency_query(window))


def prom_error_rate(client: MetricsClient, window: str) -> dict[str, float]:
    return client.query(prom_error_rate_query(window))


# Each strategy is (p95_fn, error_rate_fn). Tempo is primary/default.
P95Fn = Callable[[MetricsClient, str], dict[str, float]]
TEMPO_STRATEGY: tuple[P95Fn, P95Fn] = (tempo_p95, tempo_error_rate)
PROM_STRATEGY: tuple[P95Fn, P95Fn] = (prom_p95, prom_error_rate)


# --- Decision logic -----------------------------------------------------------

def degraded_services(
    client: MetricsClient,
    baseline_p95: dict[str, float],
    window: str,
    strategy: tuple[P95Fn, P95Fn] = TEMPO_STRATEGY,
) -> set[str]:
    """Set of services that breached SLO in the current window.

    A service is degraded iff EITHER:
      * windowed p95 latency > LAT_MULT * its baseline p95, OR
      * windowed error rate  > ERR_THRESH.

    Services absent from `baseline_p95` are not latency-evaluated (no baseline to
    compare against) but are still error-evaluated. `strategy` selects the
    transport-appropriate query builders (Tempo by default).
    """
    p95_fn, error_rate_fn = strategy
    current_p95 = p95_fn(client, window)
    error_rate = error_rate_fn(client, window)

    degraded: set[str] = set()

    for svc, p95 in current_p95.items():
        base = baseline_p95.get(svc)
        if base is not None and base > 0 and p95 > LAT_MULT * base:
            degraded.add(svc)

    for svc, rate in error_rate.items():
        if rate > ERR_THRESH:
            degraded.add(svc)

    return degraded


def capture_runs(
    client: MetricsClient,
    baseline_p95: dict[str, float],
    window: str,
    n_windows: int,
    strategy: tuple[P95Fn, P95Fn] = TEMPO_STRATEGY,
) -> list[list[str]]:
    """Sample `degraded_services` n_windows times; one sorted list per window.

    Each call is an independent measurement window -> one inner `runs` list. We
    do NOT sleep here: window spacing/scrape cadence is the caller's concern (the
    runbook), keeping this unit pure and test-friendly.
    """
    return [
        sorted(degraded_services(client, baseline_p95, window, strategy))
        for _ in range(n_windows)
    ]


def capture_baseline_p95(
    client: MetricsClient,
    window: str,
    strategy: tuple[P95Fn, P95Fn] = TEMPO_STRATEGY,
) -> dict[str, float]:
    """One-shot baseline p95 per service (used when no --baseline file is given).

    Capture this BEFORE injecting the fault so it reflects the healthy steady
    state. Returns {service: p95_seconds}.
    """
    p95_fn, _ = strategy
    return p95_fn(client, window)


# --- Output -------------------------------------------------------------------

def write_run_file(scenario_id: str, stratum: str, runs: list[list[str]], out_path) -> Path:
    """Dump the pipeline-consumed JSON record and return the written path.

    Shape is EXACTLY {"scenario_id", "stratum", "runs"} -- compatible with
    groundtruth.journey_probe.probe_file_to_ground_truth.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"scenario_id": scenario_id, "stratum": stratum, "runs": runs}
    out_path.write_text(json.dumps(record, indent=2))
    return out_path


# --- CLI ----------------------------------------------------------------------

def _load_baseline(client: MetricsClient, baseline_path: str | None,
                   baseline_window: str | None, window: str,
                   strategy: tuple[P95Fn, P95Fn]) -> dict[str, float]:
    """Resolve the baseline p95 map from either a JSON file or a live capture.

    Precedence: explicit --baseline file wins; else capture a baseline window
    live (using --baseline-window if given, else the measurement --window).
    """
    if baseline_path:
        loaded = json.loads(Path(baseline_path).read_text())
        return {str(k): float(v) for k, v in loaded.items()}
    return capture_baseline_p95(client, baseline_window or window, strategy)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m groundtruth.probe_live",
        description="Measure the degraded-service set per fault injection from "
                    "Tempo per-service RED metrics; emit the run-list JSON.",
    )
    ap.add_argument("--scenario-id", required=True,
                    help="scenario id written into the output record")
    ap.add_argument("--stratum", default="hidden",
                    help='"given" | "hidden" (default: hidden)')
    ap.add_argument("--transport", choices=["tempo", "prom"], default="tempo",
                    help="metrics transport: tempo TraceQL (primary, default) or "
                         "prom Prometheus/VM fallback")
    ap.add_argument("--tempo-endpoint", default="http://localhost:3200",
                    help="Tempo base URL for --transport tempo "
                         "(default: http://localhost:3200)")
    ap.add_argument("--endpoint",
                    help="Prometheus/VM base URL for --transport prom (e.g. "
                         "http://victoria-metrics.monitoring:8428)")
    ap.add_argument("--window", default="1m",
                    help="measurement window per run; for tempo passed as "
                         "since=<window>, for prom interpolated into the range "
                         "selector (default: 1m)")
    ap.add_argument("--windows", type=int, default=3,
                    help="number of measurement windows = number of runs (default: 3)")
    ap.add_argument("--baseline",
                    help="path to a JSON {service: baseline_p95_seconds}; if "
                         "omitted, a baseline window is captured live first")
    ap.add_argument("--baseline-window",
                    help="window for the live baseline capture (default: --window)")
    ap.add_argument("--out", required=True,
                    help="output path for the run-list JSON record")
    args = ap.parse_args(argv)

    if args.transport == "tempo":
        client: MetricsClient = TempoMetricsClient(args.tempo_endpoint)
        strategy = TEMPO_STRATEGY
    else:
        if not args.endpoint:
            ap.error("--transport prom requires --endpoint")
        client = PromMetricsClient(args.endpoint)
        strategy = PROM_STRATEGY

    baseline_p95 = _load_baseline(client, args.baseline, args.baseline_window,
                                  args.window, strategy)
    print(f">>> [{args.transport}] baseline p95 over {len(baseline_p95)} services: "
          f"{sorted(baseline_p95)}")

    runs = capture_runs(client, baseline_p95, args.window, args.windows, strategy)
    for i, run in enumerate(runs, 1):
        print(f">>> window {i}/{args.windows} degraded: {run}")

    out = write_run_file(args.scenario_id, args.stratum, runs, args.out)
    print(f">>> wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
