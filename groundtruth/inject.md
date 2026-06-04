# Live fault-injection runbook (Week-1)

Produces the 2 live ground-truth records that anchor the demo. Each scenario is
replicated 3×; a service counts as impacted if it degrades in ≥2/3 runs. Output
of each probe run is a JSON file in the `probe_runs_sample.json` shape, fed to
`groundtruth.journey_probe.probe_file_to_ground_truth`.

## Prereqs
- Online Boutique deployed to a cluster you control (the user has clusters).
  ```
  git clone https://github.com/GoogleCloudPlatform/microservices-demo data/online_boutique/repo
  kubectl apply -f data/online_boutique/repo/release/kubernetes-manifests.yaml
  ```
- A journey probe that exercises browse → add-to-cart → checkout and records,
  per request path, which downstream service exceeded the SLO (e.g. p95 > 500ms
  or error). Record the degraded set per run.

## Scenario A — EXTRA_LATENCY config coupling (the Week-1 hidden edge)
1. Baseline probe (3 runs), confirm nothing degrades.
2. Inject: `kubectl set env deployment/productcatalogservice EXTRA_LATENCY=3s`
3. Probe 3×, record degraded services per run → `data/groundtruth/live-extra-latency.json`.
4. Revert: `kubectl set env deployment/productcatalogservice EXTRA_LATENCY=0s`.

## Scenario B — CPU-saturation coupling (the state contrast)
1. Throttle: `kubectl patch deployment productcatalogservice -p \
   '{"spec":{"template":{"spec":{"containers":[{"name":"server","resources":{"limits":{"cpu":"50m"}}}]}}}}'`
2. Probe 3× under load → `data/groundtruth/live-saturation.json` (s_t = saturated).
3. Restore original CPU limits.

## Notes
- These are latency/error injections (not just scale-to-0), per §7.
- The same files load via `journey_probe.probe_file_to_ground_truth`; the eval is
  blind to whether a record came from RCAEval or a live probe.
