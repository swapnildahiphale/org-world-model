"""Live multi-pass measurement orchestrator for the OWM coupling experiment.

Companion to ``scripts/live_run.py``. Where ``live_run.py`` consumes an already
MEASURED blast set D and runs the predictor matrix offline, this script is the
thing that *produces* D against a real cluster: it bumps load, captures a healthy
baseline, then per coupling injects a single fault, lets it soak under load, and
measures the degraded set over several independent windows before reverting.

Two actions (one per invocation), mirroring the source orchestrator's structure:

  baseline   Bump the load generator, let traffic stabilise, and capture a
             per-service baseline p95 (healthy steady state) to ``--out-baseline``.

  couple     Inject one fault (``--kind env_latency`` sets EXTRA_LATENCY on the
             target deployment; ``--kind cpu`` squeezes the CPU request+limit),
             wait, measure ``--windows`` Tempo windows, then ALWAYS revert in a
             finally block and write the measurement JSON
             ``{id, origin, deployment, kind, runs, blast}``.

Degraded-set rule (per service, per window), inherited from groundtruth.probe_live:
    a service is degraded iff
        windowed p95  >  LAT_MULT * baseline_p95      (default LAT_MULT = 2.0)
      OR windowed error_rate  >  ERR_THRESH           (default ERR_THRESH = 0.01)
The baseline p95 is captured once, healthy, before any injection.

Each measurement window is an INDEPENDENT pass: non-overlapping 1-minute slices
ending ~30s in the past (to clear Tempo's ingestion/flush lag). The per-window
degraded sets are the ``runs`` list; ``blast`` is their two-thirds-majority
aggregate (groundtruth.journey_probe.aggregate_runs), the label-noise control.

SAFETY: every kubectl mutation is gated on ``--confirm-context <substring>``.
Before any mutating call we assert the substring appears in the current
kubectl context (``kubectl config current-context``); otherwise we abort. There
is deliberately no hardcoded cluster name anywhere -- the operator names the
expected context on each run.

This is cluster-only glue (like ``scripts/live_run.py`` it has no unit test).

Usage:
  # 1) capture healthy baseline
  .venv/bin/python -m scripts.cluster_run baseline \
      --namespace owm-demo --confirm-context my-cluster \
      --out-baseline /tmp/owm_baseline.json

  # 2) measure one coupling's blast radius
  .venv/bin/python -m scripts.cluster_run couple \
      --namespace owm-demo --confirm-context my-cluster \
      --baseline-file /tmp/owm_baseline.json \
      --id c1 --origin 'productcatalogservice::EXTRA_LATENCY' \
      --deployment productcatalogservice --kind env_latency --value 3s \
      --out /tmp/owm_meas_c1.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

from groundtruth.journey_probe import aggregate_runs
from groundtruth.probe_live import (
    ERR_THRESH,
    LAT_MULT,
    TempoMetricsClient,
    capture_baseline_p95,
    tempo_error_rate_numerator_query,
    tempo_p95_latency_query,
    tempo_total_rate_query,
)

# --- kubectl helpers ----------------------------------------------------------


def kc(*args: str, check: bool = True) -> str:
    """Run a kubectl command, echoing it, and return stripped stdout."""
    r = subprocess.run(["kubectl", *args], capture_output=True, text=True)
    print(f"  $ kubectl {' '.join(args)} -> rc={r.returncode}", flush=True)
    if r.returncode and r.stderr.strip():
        print("    ERR:", r.stderr.strip()[:200], flush=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def guard_context(substring: str) -> None:
    """Abort unless ``substring`` is in the current kubectl context.

    Replaces the source's hardcoded cluster guard: the operator names the
    expected context per run, and no cluster name is baked into this file.
    Call this BEFORE any mutating kubectl command.
    """
    ctx = kc("config", "current-context")
    if substring not in ctx:
        raise SystemExit(
            f"ABORT: current context {ctx!r} does not contain "
            f"--confirm-context {substring!r}"
        )
    print(f"context OK: {ctx} (contains {substring!r})", flush=True)


def pf_start(tempo_ns: str, tempo_svc: str, tempo_port: int,
             ready_pause: float = 7.0) -> subprocess.Popen:
    """Start a self-contained Tempo port-forward and pause for readiness."""
    proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", tempo_ns, tempo_svc,
         f"{tempo_port}:{tempo_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(ready_pause)
    return proc


# --- CPU squeeze (capture originals, restore on revert) -----------------------


def get_cpu(dep: str, namespace: str) -> tuple[str | None, str | None]:
    """Return the target container's (limit_cpu, request_cpu), or (None, None)."""
    lim = kc("get", f"deployment/{dep}", "-n", namespace, "-o",
             "jsonpath={.spec.template.spec.containers[0].resources.limits.cpu}",
             check=False) or None
    req = kc("get", f"deployment/{dep}", "-n", namespace, "-o",
             "jsonpath={.spec.template.spec.containers[0].resources.requests.cpu}",
             check=False) or None
    return (lim, req)


def set_cpu(dep: str, namespace: str, limit: str, request: str,
            check: bool = False) -> None:
    """Set BOTH request and limit so a low ceiling actually applies and throttles.

    Kubernetes requires request <= limit; callers pass request <= limit.
    """
    kc("set", "resources", f"deployment/{dep}", "-n", namespace,
       f"--limits=cpu={limit}", f"--requests=cpu={request}", check=check)


# --- measurement --------------------------------------------------------------


def measure_windows(client: TempoMetricsClient, baseline: dict[str, float],
                    n_windows: int) -> list[list[str]]:
    """Measure ``n_windows`` independent degraded sets.

    Each window is a non-overlapping 1-minute slice ending ~30s in the past
    (Tempo flush lag). A service is degraded in a window iff its windowed p95
    exceeds ``LAT_MULT * baseline`` OR its windowed error rate exceeds
    ``ERR_THRESH``. Returns one sorted degraded-list per window.
    """
    runs: list[list[str]] = []
    now = int(time.time())
    for k in range(n_windows):
        end = now - 30 - k * 60
        start = end - 60
        p95 = client.query(tempo_p95_latency_query(), start=start, end=end)
        errs = client.query(tempo_error_rate_numerator_query(), start=start, end=end)
        tot = client.query(tempo_total_rate_query(), start=start, end=end)
        error_rate = {s: (errs.get(s, 0.0) / t if t > 0 else 0.0)
                      for s, t in tot.items()}
        degraded: set[str] = set()
        for svc, v in p95.items():
            b = baseline.get(svc)
            if b and b > 0 and v > LAT_MULT * b:
                degraded.add(svc)
        for svc, rate in error_rate.items():
            if rate > ERR_THRESH:
                degraded.add(svc)
        runs.append(sorted(degraded))
        print(f"    window {k + 1}: degraded={sorted(degraded)}", flush=True)
    return runs


# --- actions ------------------------------------------------------------------


def do_baseline(args: argparse.Namespace) -> None:
    """Bump the load generator, stabilise, and capture a healthy baseline p95."""
    guard_context(args.confirm_context)
    proc = pf_start(args.tempo_ns, args.tempo_svc, args.tempo_port)
    try:
        kc("set", "env", "deployment/loadgenerator", "-n", args.namespace,
           f"USERS={args.loadgen_users}", check=False)
        print(f"loadgen USERS={args.loadgen_users} (light, preserves selective "
              f"blast); stabilizing {args.stabilize}s...", flush=True)
        time.sleep(args.stabilize)
        client = TempoMetricsClient(f"http://localhost:{args.tempo_port}")
        base = dict(capture_baseline_p95(client, args.baseline_window))
        with open(args.out_baseline, "w") as f:
            json.dump(base, f)
        print("baseline p95 (ms):",
              {k: round(v * 1000, 1) for k, v in sorted(base.items())}, flush=True)
        print(f"wrote baseline -> {args.out_baseline}", flush=True)
    finally:
        proc.terminate()


def do_coupling(args: argparse.Namespace) -> None:
    """Inject one fault, soak, measure N windows, then ALWAYS revert."""
    guard_context(args.confirm_context)
    with open(args.baseline_file) as f:
        baseline = json.load(f)

    proc = pf_start(args.tempo_ns, args.tempo_svc, args.tempo_port)
    orig = get_cpu(args.deployment, args.namespace) if args.kind == "cpu" else None
    print(f"### {args.id} {args.origin}: inject {args.kind}={args.value} on "
          f"{args.deployment} (orig cpu={orig}) ###", flush=True)
    try:
        client = TempoMetricsClient(f"http://localhost:{args.tempo_port}")
        if args.kind == "env_latency":
            kc("set", "env", f"deployment/{args.deployment}", "-n", args.namespace,
               f"EXTRA_LATENCY={args.value}")
        else:
            # Squeeze ceiling AND request to value so the limit applies + throttles.
            set_cpu(args.deployment, args.namespace, args.value, args.value)
        print(f"injected; waiting {args.wait}s under load...", flush=True)
        time.sleep(args.wait)
        runs = measure_windows(client, baseline, args.windows)
        blast = sorted(aggregate_runs(runs))
        out = {
            "id": args.id,
            "origin": args.origin,
            "deployment": args.deployment,
            "kind": args.kind,
            "runs": runs,
            "blast": blast,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"### {args.id} BLAST = {blast} (wrote {args.out}) ###", flush=True)
    finally:
        # Always revert the injected fault, regardless of how we got here.
        if args.kind == "env_latency":
            kc("set", "env", f"deployment/{args.deployment}", "-n", args.namespace,
               "EXTRA_LATENCY=0s", check=False)
        elif orig and orig[0]:
            # Restore the captured originals; request defaults to limit if absent.
            set_cpu(args.deployment, args.namespace, orig[0], orig[1] or orig[0])
        print(f"reverted {args.deployment}; recovery {args.recovery}s...", flush=True)
        time.sleep(args.recovery)
        proc.terminate()


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cluster_run",
        description="Live multi-pass measurement orchestrator (companion to "
                    "scripts/live_run.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared flags (declared on the parent so both subcommands inherit them).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--namespace", required=True,
                        help="namespace of the application under test")
    common.add_argument("--confirm-context", required=True,
                        help="substring that MUST appear in the current kubectl "
                             "context before any mutation (safety guard)")
    common.add_argument("--tempo-svc", default="svc/tempo-system-query-frontend",
                        help="Tempo query-frontend service to port-forward")
    common.add_argument("--tempo-ns", default="tempo",
                        help="namespace Tempo runs in")
    common.add_argument("--tempo-port", type=int, default=3200,
                        help="local + remote Tempo port for the port-forward")

    sub = ap.add_subparsers(dest="action", required=True)

    p_base = sub.add_parser(
        "baseline", parents=[common],
        help="bump loadgen, stabilise, capture healthy baseline p95",
    )
    p_base.add_argument("--loadgen-users", type=int, default=10,
                        help="USERS to set on deployment/loadgenerator")
    p_base.add_argument("--baseline-window", default="3m",
                        help="lookback window for the baseline p95 query")
    p_base.add_argument("--stabilize", type=int, default=90,
                        help="seconds to let traffic stabilise before capture")
    p_base.add_argument("--out-baseline", default="/tmp/owm_baseline.json",
                        help="path to write the captured baseline p95 JSON")
    p_base.set_defaults(func=do_baseline)

    p_cpl = sub.add_parser(
        "couple", parents=[common],
        help="inject one fault, wait, measure N windows, revert",
    )
    p_cpl.add_argument("--baseline-file", required=True,
                       help="baseline p95 JSON captured by the baseline action")
    p_cpl.add_argument("--id", required=True,
                       help="coupling id (e.g. c1), used in the output record")
    p_cpl.add_argument("--origin", required=True,
                       help="origin config node, e.g. 'svc::EXTRA_LATENCY'")
    p_cpl.add_argument("--deployment", required=True,
                       help="deployment to inject the fault into")
    p_cpl.add_argument("--kind", required=True, choices=["env_latency", "cpu"],
                       help="env_latency=set EXTRA_LATENCY; cpu=squeeze CPU req+limit")
    p_cpl.add_argument("--value", required=True,
                       help="EXTRA_LATENCY value (e.g. 3s) or CPU value (e.g. 50m)")
    p_cpl.add_argument("--windows", type=int, default=3,
                       help="number of independent measurement windows")
    p_cpl.add_argument("--wait", type=int, default=210,
                       help="seconds to soak under load before measuring")
    p_cpl.add_argument("--recovery", type=int, default=45,
                       help="seconds to wait after revert before exiting")
    p_cpl.add_argument("--out", default=None,
                       help="measurement JSON path (default /tmp/owm_meas_<id>.json)")
    p_cpl.set_defaults(func=do_coupling)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "couple" and args.out is None:
        args.out = f"/tmp/owm_meas_{args.id}.json"
    args.func(args)
    print("DONE", args.action, flush=True)


if __name__ == "__main__":
    main()
