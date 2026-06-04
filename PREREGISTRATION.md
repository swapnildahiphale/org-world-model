# Pre-registration — OWM Week-1

Committed BEFORE scoring on real ground truth (spec §3, §8). The frozen scenario
set's SHA-256 is recorded here; changing scenarios after this point invalidates
the run.

## Hypotheses
- **H1 (calibration):** on the HIDDEN stratum, the engine's reliability diagram
  tracks y=x and its Brier beats every baseline.
- **H2 (learning generalization):** after the TEACH incident, the engine's Brier
  on the held-out TRANSFER set improves; the learning-disabled clone does not;
  the RAG LLM (same incident history) does not match the engine's calibration.
- **Locality:** the lesson stays local to the config knob that caused the incident — a benign sibling knob on the same service is unaffected (a property of fine-grained credit assignment, not a learned discriminator).
- **State:** the same change yields a larger blast under `saturated` than `healthy`.

## Win conditions (Week-1)
- HIDDEN-stratum Brier(engine, post-learning) < Brier(each baseline).
- deepening: brier_post < brier_pre on TRANSFER.
- ablation: learning-disabled engine shows no deepening.
- locality: NEGATIVE predicted P < 0.1 for the coupled services (sibling knob is unaffected by construction — fine-grained credit assignment, not learned discrimination).
- state: state_delta > 0 on the coupled services.

## Frozen artifacts
- Model ID (baselines): `claude-opus-4-8`, temperature 0.7, self-consistency n=5.
- Surprise threshold: 0.1. Beta prior: Beta(1,1). New-edge prior: Beta(2,1).
- Scenario set hash (scenarios.json): `b6f90686f6e4bd693b2a74ba9e0152391977ec651f7231559547ba556d93a53a`.
- Hidden edge under test (Week-1): `pc-latency-coupling` (EXTRA_LATENCY class).

## Not in Week-1
- Abstention / risk–coverage / AURC (uses Beta-posterior variance) is specified but NOT implemented in Week-1 — it is Week-2.

## Stratification
Headline metrics on the HIDDEN stratum only. GIVEN is a demoted sanity floor and
also confirms the LLM baseline was handed sufficient context.
