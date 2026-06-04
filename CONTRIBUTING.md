# Contributing to Org World Model (OWM)

Thank you for your interest in contributing. The sections below cover how to get started, how the repo is laid out, and the non-negotiable rules that protect the integrity of OWM's evaluation.

---

## Development setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/swapnildahiphale/org-world-model.git
cd org-world-model
pip install -e ".[dev]"

# Run the test suite
pytest -q
```

All pull requests must pass `pytest -q` before review.

---

## Repository layout

| Directory / file | Purpose |
|---|---|
| `owm/` | Core engine — graph construction, causal inference, evaluation logic |
| `groundtruth/` | Ground-truth loaders — reads and validates scenario fixture files |
| `scripts/` | Entry-point runners (`run.py` exposes the `owm-run` CLI command) |
| `tests/` | Pytest test suite — unit and integration tests |
| `data/` | Static fixture data (checked in) and live-run outputs (git-ignored) |

When adding new functionality, keep the layer boundaries clean: engine logic belongs in `owm/`, I/O and orchestration in `scripts/`, fixture loading in `groundtruth/`.

---

## Evaluation integrity (non-negotiable)

OWM is a research artifact. Its headline numbers are only meaningful if they are computed honestly. Every contributor — and every PR reviewer — is responsible for upholding the two invariants below.

### Rule 1 — Headline metrics on the HIDDEN stratum only

Headline metrics (precision, recall, F1, delta vs. LLM baseline) **must be reported exclusively on the HIDDEN stratum** — the set of couplings the model was never told about and had to learn on its own.

Reporting wins on the GIVEN structure (edges handed to the model at load time) is equivalent to grading the model on its own cheat sheet. Any PR that touches evaluation output, scenario definitions, or reporting code must preserve the GIVEN / HIDDEN partition and must not aggregate or conflate the two strata in any headline figure.

### Rule 2 — Taught nodes must come from real, statically parsed manifests

Any node marked as "taught" (i.e., supplied to the engine as a known starting point) **must be a genuine node that appears in the real manifest**, recovered by static parsing. It must never be a hand-authored or synthetic identifier invented for the purpose of a test scenario.

Equally, the hidden coupling that the engine must learn must be something it genuinely has to discover — not something seeded into the graph by the scenario author. Seeding the answer and then claiming the engine found it defeats the evaluation entirely.

### Scope of these rules

PRs touching any of the following files are subject to mandatory review against both rules above, and must keep all tests green:

- `owm/eval.py`
- `owm/scenarios.py`
- Ground-truth assembly scripts or fixture files under `data/` or `groundtruth/`

If you are unsure whether a proposed change respects these invariants, open a draft PR and ask in the description. It is better to discuss early than to have a PR reverted after review.

---

## Opening issues and pull requests

- Use the issue templates for bugs and feature requests.
- Link any PR to the issue it addresses.
- Keep commits focused; prefer small, reviewable changes over large omnibus PRs.
- If your change affects published benchmark numbers, update `RESULTS.md` accordingly and explain the delta in the PR description.
