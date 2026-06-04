# Live cluster result — does the thesis hold?

**Claim under test:** a causal graph that *deepens from experience* predicts a code change's
blast radius better than a stateless LLM — by learning a runtime coupling it was never given,
and reporting calibrated confidence.

**Verdict: yes, on cluster-measured ground truth, against real Opus.** All four pre-registered
win conditions passed.

```
[PASS] engine_beats_baselines
[PASS] engine_deepens
[PASS] ablation_flat
[PASS] state_amplifies
```

---

## What was actually run (not a stub, not a fixture)

- **System under test:** Google's Online Boutique (12 microservices) deployed to an isolated
  namespace on a real EKS cluster, with a load generator driving
  live browse/cart/checkout traffic.
- **Ground truth was MEASURED, not authored.** We injected the hidden coupling for real —
  `kubectl set env deploy/productcatalogservice EXTRA_LATENCY=3s` — and read the per-service p95
  latency from distributed traces (Tempo metrics-generator, `quantile_over_time(duration,.95) by
  service`) before vs. after. The fault was then reverted.

  | service | baseline p95 | under EXTRA_LATENCY=3s | degraded? |
  |---|---:|---:|---|
  | frontend | 20.7 ms | **3568 ms** | ✅ |
  | recommendationservice | 2.3 ms | **2410 ms** | ✅ |
  | productcatalogservice | 0.1 ms | 2982 ms | (fault origin, excluded) |
  | checkoutservice | 10.7 ms | 10.4 ms | — |
  | currency / payment / email | flat | flat | — |

  **Measured blast set D = {frontend, recommendationservice}.** `checkoutservice` was *checked,
  not assumed* — it stayed flat, because the heavy productcatalog traffic flows through
  frontend's browse path, not checkout. This is the real coupling structure of the system.

- **The LLM opponents are real Opus** (`claude-opus-4-8`), at n=5 self-consistency sampling
  (confidence = agreement frequency over 5 samples, not a verbalized "I'm 80% sure"). Run via the
  Claude Code subscription — no API key consumed.
- **Static parse is faithful.** The graph is seeded from the real Online Boutique k8s manifest
  (32 nodes). `EXTRA_LATENCY` is declared in its OFF state (`"0s"`) exactly as the native
  `DISABLE_PROFILER` knob is — so the parser emits `productcatalogservice::EXTRA_LATENCY` as a
  genuine, inert, *unconnected* config node. The parser cannot know it couples to anything; that
  downstream blast is the hidden edge the engine has to **learn**.

## The result — HIDDEN stratum (the only headline; Rule 1)

Brier score, lower is better:

| predictor | HIDDEN Brier | reading |
|---|---:|---|
| **OWM engine** (after learning 1 incident) | **0.013** | best — learns + calibrates the coupling |
| Opus + RAG (handed the incident history) | 0.027 | retrieval helps a lot, but still ~2× the engine's loss |
| OWM engine (before learning) | 0.083 | ties BFS — *no better than topology until it learns* |
| BFS topology walk | 0.083 | |
| stateless Opus (no incident memory) | 0.127 | worst — cannot predict a non-topological coupling |

**Deepening:** engine 0.083 → **0.013** after observing one incident.
**Ablation (learning disabled):** 0.083 → 0.083, flat — proving the gain is the *learning*, not
the seed graph or the propagation math.
**State contrast** (state-conditioned prediction): under a saturated `s_t`, predicted blast
amplifies on exactly the coupled services — recommendationservice +0.167, frontend +0.125.

## What this does and does not show

**Does:**
- Even Opus *with retrieval over the exact incident* (0.027) does not match the deepened graph
  (0.013). Stateless Opus (0.127) is far behind. The win is real and it is on a coupling that is
  invisible to static structure.
- The advantage is entirely attributable to **learning**: pre-learning the engine is no better
  than a dumb BFS walk (both 0.083), and the learning-disabled ablation never improves. This is
  the "world model that *deepens*" vs. "stateless decoder" distinction, made falsifiable.
- The ground truth is measured from the running system, so this is not us grading our own
  homework — the coupling and its blast were discovered empirically (and one expected member,
  checkoutservice, was empirically *excluded*).

**Does not (honest limits — these are Week-2):**
- **Small sample.** One hidden coupling, 3 scored scenarios (1 hidden-transfer, 1 negative, 1
  given). This demonstrates the *mechanism* convincingly; it is not yet a k-fold result over many
  hidden edges with bootstrapped confidence intervals.
- **Partial trace coverage.** adservice/cartservice/shippingservice didn't surface under their own
  `service.name` (runtime auto-instrumentation didn't honor `OTEL_SERVICE_NAME`). They sit off the
  productcatalog call path, so they're not in the blast radius regardless — but a fuller study
  would instrument them.
- **No abstention / risk–coverage yet.** Calibration is shown via the deepening curve and the
  reliability plot (`docs/reliability.png`), not yet an abstention policy.

## Multi-pass deepening — does it keep improving over many passes? (live, 2026-06-04)

The result above folds **one** incident. The natural next question — *fold many, and watch how
the system behaves.* We re-ran on the same cluster and gathered **8 measurement windows** of the
`EXTRA_LATENCY=3s` fault under light, steady load, treating **each window as one learning pass**,
and folded them one at a time.

**Measured per-window structure (the real coupling), origin excluded:**

| service | windows degraded | rate |
|---|---:|---:|
| frontend | 8 / 8 | 1.00 |
| recommendationservice | 8 / 8 | 1.00 |
| checkoutservice | 5 / 8 | **0.62** (intermittent) |
| all others | 0 / 8 | 0.00 |

Denoised binary blast **D = {frontend, recommendationservice}** — `checkoutservice` is *below* the
2/3 majority, so it's excluded from the hard blast (the non-topological selectivity, recovered at
light load), but it is genuinely a **partial** coupling.

**Convergence over passes** (`docs/multipass_convergence.png`):

- **Per-window Brier** (the faithful probabilistic score): **0.219 (pre) → 0.041 (8 passes)** — ~5×
  — while the learning-**off** ablation stays flat at 0.219. The gain is entirely the learning.
- **Learned weights converge to the empirical rates:** frontend → 0.90, recommendationservice → 0.90
  (both climbing toward 1.0), checkoutservice → ~0.75 toward its 0.62 rate. The engine learns a
  **graded** coupling — checkout is *partially* coupled — which a topology view
  (checkout calls productcatalog ⇒ checkout is in the blast) gets wrong as a hard 1.

**Live engine vs real Opus** (k-fold harness, HIDDEN-stratum Brier vs the binary blast `D`):

| predictor | HIDDEN Brier | reading |
|---|---:|---|
| **OWM engine** (after 8 passes) | **0.067** | best — learns + calibrates the coupling |
| Opus + RAG (handed the incident history) | 0.083 | retrieval helps, still loses to the deepened graph |
| BFS / ablation / engine-pre | 0.167 | topology can't reach a disconnected config node |
| stateless Opus (no memory) | 0.250 | worst — cannot predict a non-topological coupling |

Deepening **0.167 → 0.067**; ablation **flat 0.167 → 0.167**; engine beats **all three** baselines,
including the RAG-Opus that was *handed* the incidents.

**Honest notes:**

- **Calibration vs the binary label.** Against the *collapsed* binary blast, learning checkout's
  intermittency slightly *raises* Brier (0.026 at pass 2 → 0.067 at pass 8) — the binary label
  discards the graded truth. The **per-window** score (which rewards calibration) improves
  monotonically. Both are reported; the engine wins on both vs the LLM/BFS.
- **One coupling.** Online Boutique's fault surface is thin: `EXTRA_LATENCY` exists only on
  productcatalog; CPU-limit squeezes don't bite on near-idle services (recommendation at 25m under
  light load → empty blast) and under heavy load their blast collapses to "all callers"
  (topological). So the multi-pass *depth* is on the single latency coupling, and the bootstrapped
  CI is degenerate at k = 1. Breadth across genuinely non-topological couplings needs a richer
  fault surface (e.g. an app with per-service latency knobs).
- Same two honesty rules hold: headline only on the HIDDEN stratum; the taught node
  (`productcatalogservice::EXTRA_LATENCY`) is the genuinely-parsed, inert config node.

Artifacts: `docs/multipass_convergence.png`, `docs/multipass.json`, `docs/reliability_kfold.png`,
`docs/results_kfold.json`.

## Reproduce

```
# (cluster + Online Boutique deployed, EXTRA_LATENCY measured -> D)
.venv/bin/python -m scripts.live_run \
   --degraded "frontend,recommendationservice" \
   --repo-manifest data/online_boutique/repo/release/kubernetes-manifests.yaml \
   --provider claude-code --self-consistency-n 5 --out results_live
.venv/bin/python -m scripts.check_wins results_live/results.json

# multi-pass (k couplings, within-coupling deepening): measured GT + per-window incident stream
.venv/bin/python -m scripts.run_kfold \
   --repo-manifest <staged manifest with EXTRA_LATENCY=0s> \
   --ground-truth data/groundtruth/ground_truth_kfold_live.json \
   --incidents data/groundtruth/incidents_kfold_live.json \
   --provider claude-code --out results_kfold_live
```

Committed copies: single-pass raw `docs/results.json` + `docs/reliability.png`; multi-pass
`docs/results_kfold.json`, `docs/multipass.json`, and the convergence plot
`docs/multipass_convergence.png`.
