# Org World Model (OWM) — Architecture & End-to-End Design

> **Status:** Design (productization). Extends — does not replace — the validated
> research spike (see `docs/superpowers/specs/2026-06-03-org-world-model-spike-design.md`).
> **Scope of this doc:** the architecture and end-to-end design to take OWM from a
> single-coupling research artifact to a real, runnable, scale-out system, plus a
> phased build roadmap. It does **not** build the services (those are follow-on
> tasks) and does **not** design multi-tenant SaaS specifics.
> **Date:** 2026-06-23.

---

## 0. Context & purpose

### What the spike proved
The spike validated the core thesis on a **real microservice system** (Google's
Online Boutique on a live EKS cluster), with a pre-registered protocol and
measured ground truth:

- A typed weighted causal graph answering `do(change, state) → blast radius`
  (noisy-OR propagation, state-gated) beats a stateless LLM at predicting the
  consequences of a change, **and deepens** from each incident.
- Injected a hidden latency coupling (`EXTRA_LATENCY` on `productcatalogservice`);
  measured the blast set from distributed traces = `{frontend, recommendationservice}`
  (`checkoutservice`, which topology would implicate, was empirically excluded).
- **HIDDEN-stratum Brier (lower is better):** OWM engine **0.013** < Opus+RAG
  0.027 < topology-BFS / engine-before-learning 0.083 < stateless Opus 0.127
  (real `claude-opus-4-8`, n=5 self-consistency).
- **Deepening:** `0.083 → 0.013` after one incident; learning-disabled ablation
  stays flat at 0.083 — proving the gain is the *learning*, not the seed graph.

The mechanism is convincing on one coupling. It is **not yet** a productized
system: no persistence, single-app, single learned coupling, a one-shot script,
no consumption surface.

### What this design delivers
The path from that artifact to a **single-org, scale-out** system that is the
**persistent, organization-specific causal substrate that grounds AI coding
agents** (Claude Code, Codex, and peers) during day-to-day engineering work —
feature changes, infra/config changes, debugging, implementation.

### North star (the product)
> When an engineer (or their coding agent) is about to touch the system, OWM
> answers *"given how this organization actually behaves, what is the blast
> radius of this change — what hidden couplings, which owners, which past
> incidents, and how confident are we?"* — and it gets sharper every time
> reality proves it right or wrong.

The primary **consumer is the coding agent**. OWM is an *engine*, consumed by any
agent — "sell the engine, not the seat."

### Scope & non-goals
- **In scope:** service decomposition, data model & storage, the continuous
  learning loop, ingestion, the query/consumption surface (MCP + portable rules),
  end-to-end flows, tech-stack & boundary justifications, a phased roadmap.
- **Out of scope (noted, not designed here):** actually building the services;
  multi-tenant/SaaS commercialization; re-litigating the thesis or the engine's
  math (settled by the spike).
- **Substrate-agnostic & built in public.** No organization-specific hardcoding.
  Vimo/HIX is consumer #1; OpenSRE (the author's incident-investigation platform)
  is, at most, a *future adapter* — neither is baked into the core.

---

## 1. What OWM is — and is not

This section exists because the single most common misreading is *"isn't this
just a graph database?"*

**The model = structure + parameters + engine.** Not the database; not a
code-symbol graph.

- **Structure** — a typed graph: services, config-knobs, datastores, incidents as
  nodes; `calls`, `depends_on`, `couples`, `config_affects` as edges.
- **Parameters** — **Beta distributions** on the learned edges: a probability
  *with* an uncertainty.
- **Engine** — initialization (parse), inference (`do`-propagation), and the
  update rule (learning).

A **graph DB is a map** (what connects to what; you query for neighbors /
reachability). A **world model is a predictor of consequences** (given an action
*and the current state*, what happens — and it can be wrong, checked, corrected).
Four behaviors separate them, and the spike *measured* the gap:

| predictor | what it actually is | HIDDEN-stratum Brier |
|---|---|---:|
| topology BFS over the static graph | a topology map you traverse for blast radius | 0.083 |
| the full model, **learning disabled** | the graph with no deepening | 0.083 |
| the full **world model** | interventional + state-aware + deepening + calibrated | **0.013** |

The four behaviors that produce the 6× gap — and that a graph DB structurally
lacks:

1. **Interventional, not associational** — `do(change, state)` forward-propagates
   *probabilities* (noisy-OR), not "what's adjacent." Reachability is a strictly
   weaker query.
2. **State-conditioned** — the same change has a different blast radius on a
   healthy vs. a saturated system; a static graph is identical regardless of state.
3. **Deepening from outcomes** — every incident updates edge *weights*; a
   *surprise* (judged safe, reality broke) adds a HIDDEN coupling the structure
   never had.
4. **Calibrated** — every edge is a probability *with* an uncertainty (Beta), and
   we measure whether "70%" means 70%.

**OWM is also not a code-symbol graph.** It does not store a file/function/
call-site graph of the org (that is what Sourcegraph / Serena / aider's repo-map
do). Its grain is **service / config-knob / datastore / incident**. Code is used
only to (a) map a change to the service/config it touches and (b) propose
structural edges — then the fine detail is discarded (see §6).

### Differentiation vs. prior art
A survey of the field (memory/KG servers, codebase-context engines,
impact-analysis tools) places OWM precisely:

- The closest existing thing is a **static, deterministic, LLM-free blast-radius
  tool** (e.g. RepoOps `causal_blast_radius`): forward/reverse graph traversal,
  same answer every time — **no incident learning, no confidence, no
  state-conditioning.** That proves the *consumer surface* (an agent asking "what
  breaks?") is viable.
- Memory/KG systems (official MCP memory server, Graphiti/Zep, mem0, Letta) are
  *associative* recall, not *interventional* `do()`.
- Causal-incident-GNN work with learned edge weights, calibration, and
  hidden-coupling discovery exists **in research**, but is not packaged as an
  org-specific grounding service for a coding agent.

**OWM's empty niche:** an org-specific, **interventional, state-conditioned,
self-correcting, calibrated** causal model that learns from the org's own
incident history and serves calibrated blast-radius grounding to a coding agent.
The differentiator to lead with is therefore **causal + learned + calibrated +
state-aware** — *not* "blast radius" (which already exists in static form).

---

## 2. Consumption model — grounding a coding agent

### The grounding pack
The unit of consumption is a **grounding pack**, returned for "I'm about to do X":

- **Ranked, calibrated blast radius** — services/components at risk, in order,
  with probabilities and uncertainty.
- **Hidden couplings** relevant to the change (learned, not structural).
- **Ownership** of the affected services.
- **Related past incidents.**
- **Abstain / ask-a-human flag** when confidence is low (roadmap).
- **Optional prose narration** (LLM) — never on the blocking path.

### How an agent actually invokes it (the hard part)
Research across Claude Code, Codex, Cursor, Cline, Windsurf and the MCP spec
converges on one fact:

> **No MCP client supports server-driven automatic invocation.** Tools are
> *model-controlled*. The spec has no "always consult me" annotation, no server
> push (`resources`/`prompts`/annotations cannot force auto-injection). So a bare
> MCP tool *will be ignored* on the changes that matter most — the single
> most-documented failure mode in the ecosystem.

Proactivity is therefore won by **moving the "should I consult OWM" decision out
of the model's judgment**, two ways:

| | Mechanism | Reliability | Reach |
|---|---|---|---|
| **Portable (probabilistic)** | Trigger-rich tool descriptions ("call BEFORE any edit/infra change") + **always-on rules files**. Universal artifact: **`AGENTS.md`** (read by Codex, Cursor, Cline; Windsurf reads root `AGENTS.md`). Plus per-agent always-on rules (Cursor `.cursor/rules/*.mdc` `alwaysApply`, Cline `.clinerules`, Windsurf `.windsurf/rules/*.md` `always_on`). | High-probability, not guaranteed | **Every agent** |
| **Deterministic (forcing function)** | A **`PreToolUse` hook on edit/write/deploy** that intercepts the change, calls OWM, injects the grounding pack as `additionalContext`, and can *block* when risk×confidence crosses a threshold. "Hooks guarantee behavior; prompts suggest it." | Guaranteed | **Claude Code only** (no hook equivalent in Codex/Cursor/Cline/Windsurf today) |

**Decision:** the MVP ships the **portable** path (MCP server + `AGENTS.md`/rules
kit), engine-agnostic. The **deterministic Claude Code hooks** are a clearly
marked *later* phase (the "forcing-function upgrade"), not part of the first cut.
A user-invokable skill / MCP prompt (`/owm-check`) is an explicit escape hatch.

**Tool surface is deliberately tiny** (~3 tools; the "≤8 tools or they get
ignored" lesson): `owm_ground(change, state)`, `owm_blast_radius(change, state)`,
`owm_why_coupled(a, b)`, plus `owm_record_outcome(...)` for closing the loop.

---

## 3. System architecture — the spine

Everything decomposes along **write / update / read** around a central model
store, with a hard line between the **agent-agnostic engine** and the
**per-agent integration kit**. Validated spike code (`graph`, `propagate`,
`learn`, `eval`) is promoted from scripts to services — not rewritten.

```
                       SIGNAL SOURCES
   ┌───────────────┬──────────────────┬─────────────────┬──────────────────┐
   │ registered    │ k8s manifests    │ incidents:      │ unstructured:    │
   │ service repos │ + traces (Tempo) │ alert/SLO,      │ postmortems,     │
   │               │                  │ manual declare  │ Slack, ADRs      │
   └──────┬────────┴────────┬─────────┴───────┬─────────┴────────┬─────────┘
          │ BATCH           │ BATCH           │ EVENT            │ (later)
          ▼                 ▼                 ▼                  ▼
 ┌─────────────────────────────────────┐  ┌────────────────────────────────┐
 │ ① INGESTION WORKER (propose host)   │  │ ⑤ OUTCOME RECEIVER (decide host)│
 │   code-parse · topology · config    │  │   trigger → measure blast set   │
 └──────────────────┬──────────────────┘  └───────────────┬────────────────┘
        propose      │                         outcome      │
        structure    ▼                         events       ▼
 ┌─────────────────────────────────────┐        ┌──────────────────────────┐
 │ ② MODEL STATE — POSTGRES (+ queue)  │◄───────┤ ⑥ LEARNING WORKER        │
 │   nodes · edges (GIVEN/HIDDEN) ·    │ grant/ │   Beta update +          │
 │   Beta params · incident log ·      │ revoke │   surprise-gated HIDDEN   │
 │   ownership idx · model versions    ├───────►│   edges (continuous)     │
 └──────────────────┬──────────────────┘ load   └──────────────────────────┘
                    │ load graph → memory
                    ▼
 ┌─────────────────────────────────────┐
 │ ③ INFERENCE ENGINE [reuse]          │   do(change, state) → noisy-OR,
 │   in-mem graph · state-gated        │   state-conditioned, calibrated
 └──────────────────┬──────────────────┘
                    ▼
 ┌─────────────────────────────────────┐   ┌──────────────┐  ┌──────────┐
 │ ④ QUERY / GROUNDING API [new]       │──►│ MCP server   │  │ Inspect  │
 │   assembles the "grounding pack"    │   │ ~3-4 tools   │  │ UI       │
 └─────────────────────────────────────┘   └──────┬───────┘  └──────────┘
                                                   │ MCP
                                                   ▼
                       ┌───────────────────────────────────────┐
   AGENTS.md +         │  CODING AGENT — Claude Code · Codex    │
   always-on rules ───►│  (CC PreToolUse hooks = later phase)   │
   (integration kit)   └──────────────────┬────────────────────┘
                                           │ acts on grounding; ships change;
                                           └──► reality → measured outcome ──► ⑤
   off the hot path:  ⑦ OFFLINE EVAL HARNESS (CI honesty gate)
```

Legend: `[reuse]` = validated spike code promoted to a service · `[new]` =
productization work · later-phase items are named inline.

### The six components
1. **Ingestion / adapter layer** *(new wrappers + `codegraph`/`topology` reuse)* —
   structure adapters run **batch**; the code-parse adapter collapses repos *up*
   to a **file→service ownership index + config-knob inventory** (no symbol
   graph); the topology adapter seeds GIVEN edges. The outcome adapter runs
   **event-driven** (trigger → measure). All emit a normalized envelope.
2. **Model state — Postgres** *(new)* — system of record for nodes, edges (with
   provenance), Beta params, the incident log, ownership index, and model
   versions/snapshots. Closes the spike's "rebuilt each run" gap (§4).
3. **Inference engine** *(reuse `propagate`/`engine`)* — loads the persisted graph
   into memory, serves `do(change, state)`. Stateless compute → scales
   horizontally.
4. **Serving** *(new)* — the Query/Grounding API (assembles the pack), a tiny MCP
   server over it, and a read-only inspection UI.
5. **Learning service** *(reuse `learn`)* — consumes outcomes continuously; Beta
   updates + surprise-gated HIDDEN-edge proposals; bumps the model version.
6. **Offline eval harness** *(reuse `eval` + baselines)* — out of the live path;
   the honesty gate + regression detector in CI.

Plus the **integration kit** (shipped `AGENTS.md` + always-on rules; CC hooks
deferred) — artifacts, not a running service.

---

## 4. Data model & storage (component ②)

### The principle: event-sourced truth
> The **durable** source of truth is **(a) GIVEN structure** + **(b) the
> append-only incident/outcome log**. The **learned Beta weights are *derived***
> by replaying the log through the learning rule. The model is therefore both
> **reproducible** (replay → identical weights, every time → the honesty
> guarantee) **and** **fast to load** (materialized snapshots per version).
> Load-and-continue and rebuild-from-scratch yield the same model.

This is how the README's "not persisted today / rebuilt each run" gap is closed
*without* sacrificing reproducibility.

### Schema (ER)
```
                 ┌───────────────────────────────────────────────┐
                 │ node                                           │
                 │  id · type{service|config_knob|datastore|      │
                 │  incident} · key · attrs(jsonb) ·              │
                 │  first_seen · last_seen · provenance · env     │
                 └──────────────┬──────────────────┬─────────────┘
                          src ◄─┘                  └─► dst
                 ┌───────────────────────────────────────────────┐
                 │ edge                                           │
                 │  src→dst · type{calls|depends_on|couples|      │
                 │  config_affects} · provenance{GIVEN|HIDDEN} ·  │
                 │  beta_alpha, beta_beta  ← weight + uncertainty │
                 │  state_mask           ← state-conditioning     │
                 │  status{active|candidate|dormant} ·            │
                 │  evidence(jsonb→incidents) ·                   │
                 │  model_version_introduced                      │
                 └───────────────────────────────────────────────┘

  ── append-only EXPERIENCE (the durable memory) ──────────────────────────
  ┌─────────────────────────────┐  reconcile  ┌─────────────────────────────┐
  │ incident / outcome log      │◄───────────►│ prediction log              │
  │  declared_at · trigger_src ·│  (→surprise)│  made_at · change_ref ·     │
  │  window · change_ref ·      │             │  predicted_blast(ranked) ·  │
  │  observed_blast_set ·       │             │  state · model_version ·    │
  │  state_at_time · env ·      │             │  grounding_pack             │
  │  raw_signals · schema_ver   │             │  · schema_ver               │
  └──────────────┬──────────────┘             └─────────────────────────────┘
                 │ replay through learning rule → derives Beta weights
                 ▼
  ┌─────────────────────────────┐             ┌─────────────────────────────┐
  │ model_version              │             │ ownership index             │
  │  id · parent · created_at ·│             │  repo · path_glob → service │
  │  rule_version · summary ·  │             │  source{manifest|codeowners │
  │  metrics(jsonb) · snapshot │             │  |convention|image} · conf  │
  └─────────────────────────────┘             └─────────────────────────────┘
```

### What this buys
- **Model versioning / rollback** — `model_version` is a parent-linked DAG; each
  learning update bumps a version with its metrics. Rollback = repoint the serving
  projection at an earlier version. Also yields the **deepening curve over time**
  for free.
- **Prediction ⇄ outcome reconciliation** — every grounding answer is recorded
  with its `model_version`. When an outcome later lands for a matching change,
  measured-vs-predicted divergence **is the surprise signal** (§8).
- **The in-memory graph is a projection**, not a separate truth — the inference
  engine loads the latest `model_version`. The graph is small (~thousands of
  nodes), so a full in-memory load is trivial; Postgres owns durability and
  concurrency.

### Storage choice — why Postgres, not a graph DB
The hot operation is **noisy-OR propagation with state masks + Beta sampling** —
not a graph-DB traversal primitive (not expressible as Cypher) — and the
Beta/version/log tables are inherently relational. Postgres serves all of it; the
graph fits in RAM for traversal. A native graph DB (Neo4j/Memgraph) would split
the truth and buy traversal that isn't needed. **Revisit only if the graph ever
outgrows memory — not a single-org concern.** The substrate remains swappable by
design.

---

## 5. Schema evolution & extensibility

OWM's whole promise is "new signals plug in and the model grows richer **without
the core changing**." That requires schema evolution to be cheap.

### The mechanism: stable skeleton + open vocabulary + type registry + versioned log
```
   ┌───────────────────────────────────────────────────────────────┐
   │ TYPE REGISTRY   (data/config — adapters register new types)    │
   │   per node-type:  role = causal | structural | annotation      │
   │   per edge-type:  propagates? · carries Beta? · proposable-by   │
   │   → the engine ASKS the registry; it never hardcodes the list  │
   └───────────────────────────────┬───────────────────────────────┘
   ┌───────────────────────────────┴───────────────────────────────┐
   │ OPEN VOCABULARY (jsonb)          STABLE SKELETON (columns)      │
   │  node.attrs / edge.attrs    +    node(id,type,key,env) ·       │
   │  type-specific keys              edge(src,dst,type,provenance,  │
   │  → new type / new key: NO DDL    beta,state_mask,status,ver)    │
   └───────────────────────────────────────────────────────────────┘
   ┌───────────────────────────────────────────────────────────────┐
   │ IMMUTABLE EVENT LOG  +  READ-TIME UPCASTERS                     │
   │  every record stamped schema_version; history NEVER rewritten;  │
   │  replay applies v1→v2→…→current upcasters → determinism intact  │
   └───────────────────────────────────────────────────────────────┘
```

### Evolution scenarios
| Scenario | What happens | Cost |
|---|---|---|
| **Add a node type** (`team`, `queue`, `feature_flag`) | New registry entry declaring its **role** (`service`/`config_knob`/`datastore` are *causal*; `team`/`owner`/`doc` are *annotation* — surfaced in the pack, never propagated). | No DDL, no core change |
| **Add keys to a node** | Goes in `attrs` jsonb; old nodes tolerate absence. | Additive, zero migration |
| **Add an edge type** (`shares_resource`, `documented_in`) | Registry entry declaring **causal vs structural-only**; propose/decide governs weight. | No DDL, no core change |
| **Rename / change meaning of a type** | History is immutable → handled by **upcasters** at replay; a *semantic* change is a **new type**, never a silent redefinition. | One upcaster fn |
| **Adapter emits a new shape** | The **adapter envelope** is the versioned, validated contract; core validates against schema version. | Adapter evolves independently |
| **Genuine relational change** (index/table/column) | Standard **Alembic** migration; rare by design. | Normal migration |
| **Change the learning rule** | Stamp `rule_version` on `model_version`; re-derive weights by replaying the immutable log under the new rule → a new version branch. | Version branch, no data loss |

### The two invariants this protects
1. **Replay-determinism (honesty)** — history append-only; weights derived, never
   hand-edited; upcasters keep old logs replayable. Calibration/deepening claims
   stay reproducible across schema changes.
2. **Core-stability (extensibility)** — adding signals/types/edges touches *only*
   the registry + a new adapter, never the engine, storage skeleton, or query API.

**Compatibility posture:** older engine + newer data → unknown types load as
**inert annotations** (warn, don't crash); newer engine + older data →
**upcasters** fill defaults. Additive-by-default; deprecate, never delete.

---

## 6. Ingestion & the adapter model

### One interface, two hosts
Every signal source is a **plugin behind one interface**
(`source → normalized envelope`: `edge-proposal | edge-outcome | node-annotation`).
That single interface is the extensibility seam. At runtime the plugins are hosted
by **two processes, split by path** (because propose and decide have different
invocation models):

```
        ONE ADAPTER INTERFACE   (source → normalized envelope)
                 │
      ┌──────────┴────────────────────────────────────┐
      ▼                                                ▼
  ① INGESTION WORKER                            ⑤ OUTCOME RECEIVER
   batch · pull · scheduled                      event · push · webhook
   ("signals PROPOSE structure")                 ("outcomes DECIDE weight")
   plugins:                                       plugins:
    • code-parse   (repos → file→service)          • incident/alert  (PagerDuty/
    • topology     (k8s manifests + Tempo)           SLO webhook / manual declare)
    • config       (config files / secret mgr)     • trace-measure   (Tempo →
    • [later] LLM extraction                         observed blast set over window)
      (postmortems / Slack / ADRs)
   → emit PROPOSE envelopes                       → emit DECIDE (outcome) envelopes
```

- A single *source* can appear in both hosts: **Tempo** feeds the *topology*
  adapter (batch, discover GIVEN call edges) **and** the *trace-measure* adapter
  (event, measure the blast set during an incident window).
- Each adapter is a **plugin within its host**, not its own deployable — but the
  shared interface lets a heavy one (e.g. code-parse over a giant monorepo) be
  peeled into a standalone process later, purely as a deployment choice.

### The core principle: *signals propose, outcomes decide*
Structural and decision signals (code, topology, config, and later Slack/ADRs/
postmortems) may **propose** nodes/edges. **Only outcome signals** (incidents,
SLO breaches, fault injection) **grant or revoke causal weight.** This is the rule
that keeps OWM a world model and not "RAG over the wiki."

### The code graph — grain & scope (resolved)
- **Grain:** nodes are **service / config-knob / datastore / incident**. The
  code-parse adapter **collapses code *up* to the service grain** — it produces a
  **file→service ownership index** and a **config-knob inventory**, then discards
  the fine detail. No symbol graph is stored (that is another tool's job).
- **Scope:** only **repos that own deployed services** (the causally-relevant
  set), tracked in a **registry** and parsed on merge/deploy/schedule. The
  registry can be auto-populated from runtime topology (deploy image → source
  repo) or curated. Library/tooling repos enter only if/when they become causally
  relevant. *Not* "every repo in the org."
- **Cross-service call edges** are better sourced from **runtime traces** than
  static polyglot call-parsing; static parse is a weaker secondary source. The
  durable code-derived artifacts are the ownership index + config-knob inventory.

### Hybrid ingestion (batch + event)
- **Structure** = **batch** (slow-changing; idempotent reconcile on
  merge/deploy/schedule).
- **Outcomes** = **event** (time-sensitive but low-volume → a simple durable
  queue, *not* Kafka).
- **Re-derivation/backfill** = **batch** (rare; reuses the same learning-rule
  code).

---

## 7. Inference & the grounding query (read path)

### How the agent describes "the change I'm about to make"
The agent supplies any of: a **diff/PR**, a **file path**, or a **service/config
name**. The Query API **localizes** it to graph node(s) via the **ownership
index** (file/path → service) and the config-knob inventory. Localization is the
*only* place code mapping is load-bearing at read time.

### The query
`do(change, state)` runs over the in-memory projection: locate the change →
forward-propagate noisy-OR over weighted, state-masked edges → a **ranked,
calibrated blast radius**. State (`healthy | saturated | …`) is supplied by the
caller or inferred from current telemetry (roadmap).

### The grounding pack (returned to the agent)
```
  grounding_pack = {
    blast_radius:  [ {service, p, uncertainty, via:[edges]} … ],  # ranked
    hidden_couplings: [ {a, b, why, confidence} … ],
    ownership:     [ {service, owner/team} … ],
    related_incidents: [ {id, when, summary} … ],
    abstain:       {should_ask: bool, reason} ,                   # roadmap
    narration:     "<optional LLM prose>",                        # non-blocking
    model_version: <id>
  }
```
The structured pack returns immediately; **LLM narration is optional and never
blocks** the structured answer. Every pack is written to the **prediction log**
with its `model_version` to enable later reconciliation (§8).

### MCP tools (the tiny surface)
- `owm_ground(change, state)` → full grounding pack (the default).
- `owm_blast_radius(change, state)` → ranked calibrated set only.
- `owm_why_coupled(a, b)` → explanation + evidence for a coupling.
- `owm_record_outcome(change, observed, state)` → closes the loop (also reachable
  via ⑤).

LLM role across the system: **extract** (ingestion edge: unstructured → proposed
elements) + **serve/explain** (query edge: narration) + **offline baseline**.
**Never in the prediction path** — that stays graph-native (the thesis).

---

## 8. The learning / deepening loop at scale (component ⑥)

The spike ran learning as a one-shot script over a fixed incident set.
Productizing makes it a **continuous, event-driven service** with overfitting
guards, state-conditioning, and a way to forget.

```
  outcome event  (measured blast set + state_at_time + change_ref + env)
        │
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ RECONCILE                                                      │
  │  fetch prediction @ the live model_version (or compute the     │
  │  counterfactual if no agent queried it) —                      │
  │  SURPRISE := divergence(measured blast  vs  predicted blast)   │
  └───────────────┬───────────────────────────┬──────────────────┘
     low surprise │                            │ high surprise
     (confirms /  │                            │ (broke something the
      denies)     ▼                            ▼  model called safe)
   ┌──────────────────────────┐    ┌────────────────────────────────┐
   │ BETA UPDATE              │    │ STRUCTURAL PROPOSE             │
   │ existing edge(s):       │    │ generate candidate HIDDEN      │
   │ α/β += outcome,         │    │ coupling(s) — shared datastore,│
   │ CONDITIONED ON STATE    │    │ shared config knob, temporal   │
   │ (refines state_mask)    │    │ co-occurrence in traces —      │
   │                         │    │ add as status=candidate with   │
   │                         │    │ WIDE Beta (low confidence)     │
   └───────────┬─────────────┘    └───────────────┬────────────────┘
               │                                  │ subsequent outcomes
               │                                  │ CONFIRM → status=active,
               │                                  │ earns weight (or never →
               │                                  │ stays candidate/dormant)
               ▼                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ WRITE Postgres · append immutable log · bump model_version    │
   └───────────────┬──────────────────────────────────────────────┘
                   ▼
   inference engine HOT-RELOADS its projection → next grounding is deeper
```

### Propose / decide, applied to learning itself (the overfitting guard)
A **single surprising incident never creates a confident coupling.** Surprise
only **proposes a candidate** HIDDEN edge with a *wide* Beta; **subsequent
outcomes decide** whether it earns weight. Real couplings sharpen with repeated
incidents; flukes stay dormant and never pollute predictions. This is the same
propose/decide rule, one level up.

### Other at-scale properties
- **State-conditioning is in the update** — outcomes carry `state_at_time`, so a
  Beta update refines the **state-conditional** weight (`state_mask`), not a
  global one. The "money shot" becomes a learned property.
- **Continuous-service mechanics** — outcome → durable queue → learning worker →
  reconcile → update/propose → write + version-bump → engine atomically hot-swaps
  the in-memory projection (reads always see one consistent version). **Ordering &
  idempotency** come free from the event log (fold the log forward; dedupe by
  outcome id). Periodic **snapshots** so the engine doesn't replay the whole log
  on boot.
- **Batch vs streaming, sized to reality** — incidents are *rare* (tens–hundreds/
  month for one org), so live deepening is **streaming over a simple durable
  queue** (Postgres-backed `SKIP LOCKED`, or Redis/SQS) — no Kafka/Flink. Only
  **re-derivation** is batch.
- **Structural governance (hybrid).** Routine surprises **auto-add** a candidate
  edge (low confidence, hands-off). **High-surprise / high-impact** hypotheses are
  queued for a quick **human confirm** before becoming trusted. Best of both:
  automatic deepening, with a human gating the big causal claims (which also
  builds trust early).
- **Forgetting (designed-in now, implemented later).** Couplings disappear when
  code is refactored. Two invalidation paths (borrowing bi-temporal ideas from
  Zep/Graphiti): (1) **outcome-driven down-weighting** (β accumulates when a
  coupling stops firing) and (2) **structure-change invalidation** (the structure
  adapter sees the shared substrate removed → the edge goes `dormant`). Dormant ≠
  deleted — provenance/history kept (honesty). The schema's `status` field and
  immutable log support this from day one; the mechanism is built in a later phase.
- **Honesty in production.** The offline eval harness re-runs on a schedule
  (k-fold over the accumulated log) as a **regression gate**: if a learning change
  degrades held-out Brier, flag it. The honesty property stays *live*.

---

## 9. Service decomposition & boundaries

**Governing rule:** don't rewrite the validated math. The core (`graph`,
`propagate`, `learn`, `eval`) stays **Python** (networkx/numpy); services are
deployment wrappers. FastAPI + pydantic (the pydantic models *are* the adapter
envelopes and MCP `outputSchema`), official Python MCP SDK, Postgres
(jsonb + Alembic).

### The 7 deployables (target / scale-out)
```
                       ┌──────────────────────────────────────┐
  coding agent ──MCP──►│ ① MCP SERVER (thin)                  │
  (CC / Codex)         │   local stdio  OR  remote HTTP       │
                       └───────────────┬──────────────────────┘
                                       │ HTTP
  human ──browser──┐                   ▼
                   │   ┌────────────────────────────────────────┐
                   └──►│ ② READ SERVICE  (sync · low-latency · Nx)│
                       │   Query/Grounding API                   │
                       │   + EMBEDDED inference engine           │
                       │   (in-mem graph projection, noisy-OR)   │
                       └───────────────┬────────────────────────┘
                                       │ read latest model_version
                                       ▼
                       ┌────────────────────────────────────────┐
                       │ ③ POSTGRES (nodes/edges/Beta/log/       │◄───┐
                       │   versions/ownership) + QUEUE           │    │ write +
                       └──────▲────────────────────────┬─────────┘    │ version
                       batch  │ write          outcome  │ events       │ bump
                  ┌───────────┴─────────┐      ┌────────▼──────────┐   │
                  │ ④ INGESTION WORKER  │      │ ⑤ OUTCOME RECEIVER│   │
                  │   structure adapters│      │   webhook + trace-│   │
                  │   (code→service,    │      │   measure → emit  │   │
                  │   topology, config) │      │   outcome events  │   │
                  └─────────────────────┘      └────────┬──────────┘   │
                                                        │ queue        │
                                               ┌────────▼──────────┐   │
                                               │ ⑥ LEARNING WORKER │───┘
                                               │   reconcile→Beta /│
                                               │   propose HIDDEN /│
                                               │   version bump    │
                                               └───────────────────┘
  off-runtime:  ⑦ OFFLINE EVAL HARNESS (CI)  ·  INTEGRATION KIT (AGENTS.md/rules)
```

| # | Service | Sync/async | Scaling | Origin |
|---|---|---|---|---|
| ① | **MCP server** (thin; ~3-4 tools) | sync | stateless; local-stdio *or* remote-HTTP | new |
| ② | **Read service** = Query/Grounding API **+ embedded inference engine** | **sync, low-latency** | horizontal (each replica loads the projection) | new + `propagate`/`engine` reuse |
| ③ | **Postgres + queue** | — | managed; queue Postgres-backed → Redis/SQS only if needed | new |
| ④ | **Ingestion worker** (structure adapters) | **async, batch** | one worker, pluggable tasks; split per-source later | new + `codegraph`/`topology` reuse |
| ⑤ | **Outcome receiver** (trigger→measure) | async, event | thin webhook + trace measurement | new |
| ⑥ | **Learning worker** | **async, event-driven** | single consumer (ordered log-fold) | new + `learn` reuse |
| ⑦ | **Offline eval harness** | CI job | not a runtime service | reuse `eval` + baselines |

### Boundary justifications
- **Engine embedded in the read service, not a separate hop** — the agent waits
  mid-task; an extra network hop costs latency for no benefit (the graph fits in
  every replica). One sync read service.
- **MCP is a thin separate adapter** — agents consume MCP as a **local stdio**
  process next to them *or* a **remote HTTP** server; keeping it thin lets the
  same Query API serve both shapes and the human UI.
- **Sync/async maps onto read/write/update** — read = synchronous sub-second
  (structured pack returns immediately; LLM narration never blocks); write +
  update = async background workers decoupled by the queue.
- **The collapse is a deployment choice, not a re-architecture** — all runtime
  pieces share one Python core lib + one Postgres, so the MVP runs as **one
  process** (API+MCP+engine) + a **CLI** (batch ingest, manual outcomes), peeling
  off workers as volume grows (§12).

### Multi-env & access control
- **Multi-cluster/multi-env** — nodes/state/outcomes carry an `env` tag; the model
  is **env-conditioned** (a coupling can be prod-only). MVP scopes to **one env
  (prod)**; multi-env is additive thanks to the registry+jsonb design.
- **Access control** — grounding data is org-internal (topology + incident
  metadata), not secrets. MVP = **API token + MCP inside the trust boundary**;
  authn/z at the gateway, fine-grained RBAC deferred.

---

## 10. End-to-end flows

### E2E #1 — Agent grounds itself before a change (read path)
1. Agent is about to edit `productcatalogservice` (file/diff/service). Per the
   `AGENTS.md`/rules kit, it calls `owm_ground(change, state)`.
2. MCP server ① → Read service ②. The Query API **localizes** the change via the
   ownership index → `productcatalogservice` node.
3. Embedded engine ③ runs `do(change, state)` over the in-memory projection →
   ranked calibrated blast radius; the API assembles the grounding pack (couplings,
   ownership, related incidents, abstain flag, optional narration).
4. Pack returned to the agent; **written to the prediction log** with the live
   `model_version`. The agent incorporates it before acting.

### E2E #2 — An incident deepens the model (update path — the moat)
1. An alert/SLO breach fires (or a human declares an incident). The Outcome
   receiver ⑤ records the trigger and **measures the observed blast set** from
   traces over the incident window; emits an outcome event to the queue.
2. Learning worker ⑥ **reconciles** the measured set against the stored prediction
   for that change/state → computes **surprise**.
3. Low surprise → **Beta update** on existing edges (state-conditioned). High
   surprise → **propose a candidate HIDDEN edge** (wide Beta); if high-impact,
   route to the human review queue (hybrid governance).
4. Write to Postgres, append the immutable log, **bump model_version**; the read
   service hot-reloads its projection.
5. **Next time** any agent touches anything near that coupling, E2E #1 already
   reflects it. The agent's own shipped change, and what broke or held, becomes
   the next outcome — closing the loop.

### E2E #3 — Structure reconcile (write path)
1. A merge/deploy (or schedule) triggers the Ingestion worker ④.
2. The code-parse adapter re-parses registered service repos → updates the
   ownership index + config-knob inventory; the topology adapter refreshes GIVEN
   edges from manifests + traces.
3. Adapters emit **propose** envelopes; the model state ② is reconciled
   idempotently (GIVEN structure updated; learned weights untouched). Structure
   changes can trigger **forgetting** (a removed substrate → dormant edge) in a
   later phase.

---

## 11. Tech stack (summary)

| Concern | Choice | Why |
|---|---|---|
| Core language | **Python ≥3.10** throughout the backend | Don't rewrite validated math (networkx/numpy) |
| API | **FastAPI + pydantic** | async; pydantic = adapter envelopes + MCP `outputSchema` |
| Agent interface | **Official Python MCP SDK** + `AGENTS.md`/rules kit | portable across Claude Code/Codex/Cursor/Cline/Windsurf |
| Store | **Postgres** (jsonb skeleton, Alembic) | relational Beta/log/version + small in-RAM graph |
| Queue | **Postgres-backed** (`SKIP LOCKED`) → Redis/SQS if needed | incidents are low-volume; no Kafka |
| In-memory graph | **networkx** (reuse) | proven; fine at single-org scale |
| Inspection UI | minimal (server-rendered or small SPA) | secondary; can be deferred |
| Deploy | containers; k8s-native but generic | author runs EKS; design stays substrate-agnostic |
| LLM | extraction (ingest) + narration (serve) + offline baseline | never in the prediction path |

---

## 12. Phased roadmap

The thesis is proven, so phasing is "make the proven engine grounded, persistent,
and queryable by my agent on my real system" first, then automate and scale.
Every phase reuses the spike core; nothing is throwaway; the 7 deployables start
**collapsed** and peel apart as volume justifies.

### The smallest first slice — MVP "walking skeleton"
**One Python process (API + MCP + embedded engine) + a CLI + Postgres.**
- **Persist** the event-sourced model (minimal schema: nodes/edges/Beta/incident
  log/ownership/model_version) — the one big *new* thing; closes the persistence
  gap.
- **Ingest structure** for **1–few registered service repos**: code-parse →
  ownership index + service/config-knob nodes; k8s manifests → GIVEN edges (reuse).
- **Read service**: embed the engine; serve the grounding pack via `ground` /
  `blast_radius` (reuse).
- **MCP server** (~3 tools) + **`AGENTS.md`/rules kit** — makes it work inside
  Claude Code/Codex.
- **Manual outcomes**: a CLI/API to declare an incident + observed blast set →
  surprise-gated deepening + Beta update (reuse).
- **One env (prod).**

> Why this is the right smallest slice: it delivers the *entire loop* — ingest →
> agent gets grounded → record outcome → model deepens — on the user's **real**
> system, and it includes **manual deepening** (without it, an MVP would just be a
> static blast-radius tool, i.e. RepoOps, not OWM). The first commit-worthy
> increment within MVP is *persist + read + MCP* (grounding works day one);
> manual-outcome deepening lands immediately after.

### Phases
```
 MVP  walking skeleton — 1 process + CLI + Postgres ─────────────────────────
   persist (event-sourced) · 1–few repos structure · read svc (engine embedded)
   · MCP + AGENTS.md/rules · MANUAL outcome → deepen · one env
   ▸ value: my coding agent is grounded on my real system — and it deepens
 ───────────────────────────────────────────────────────────────────────────
 P1  automate the outcome feed ──────────────────────────────────────────────
   ⑤ Outcome receiver: alert/SLO webhook + Tempo trace-measure (trigger→measure)
   ▸ value: deepening with zero manual effort — the moat goes self-serve
 P2  continuous learning + governance ───────────────────────────────────────
   ⑥ Learning worker (event/queue) · HYBRID structural review queue ·
   model versioning/rollback surfaced
   ▸ value: true continuous deepening at frequency; trust controls on big claims
 P3  scale-out + visibility ─────────────────────────────────────────────────
   multi-repo registry (auto-populate from topology) · ④ ingestion worker split ·
   read service horizontal · inspection UI (graph · provenance · deepening curve)
   ▸ value: large-org scale; humans can see and trust the model
 P4  proactivity + breadth ──────────────────────────────────────────────────
   CC PreToolUse/UserPromptSubmit HOOKS (forcing function) · LLM extraction
   (postmortems/Slack/ADRs) · multi-env conditioning · forgetting/invalidation ·
   abstain/ask in the pack
   ▸ value: deterministic grounding; richer signals; staleness handled
 P5  honesty-at-scale + hardening (ongoing) ─────────────────────────────────
   scheduled k-fold eval regression gate (CI) · AURC/abstention · access-control/RBAC
   · hardened multi-agent rule kits · OSS "engine, not the seat" packaging
```

This sequences directly from the decisions in this doc: portable consumption is
in the MVP, **hooks deferred to P4**; **hybrid governance at P2**; **forgetting
designed-in-now, built P4**; **LLM-extraction + multi-env at P4**;
honesty-at-scale eval gate at P5. The collapse→peel deployment path means no
rewrites between phases.

---

## 13. Decision log

Every load-bearing decision made while designing this, with rationale:

1. **OWM is fully independent; OpenSRE is at most a future adapter.** Keeps the
   core generic / built-in-public; avoids coupling to one platform.
2. **Primary consumer = the coding agent (Claude Code/Codex).** OWM is the
   persistent, org-specific causal substrate stateless agents lack — that is the
   product, not a tagline.
3. **Consumption = MCP server + portable `AGENTS.md`/rules kit; hooks deferred.**
   MCP can't force invocation; portable rules are the cross-agent forcing path;
   deterministic CC hooks are a later upgrade. "Sell the engine, not the seat."
4. **Differentiator = causal + state-conditioned + deepening + calibrated**, not
   "blast radius" (static blast-radius already exists).
5. **Graph store = Postgres (SoR) + in-memory projection**, not a graph DB —
   noisy-OR isn't Cypher; Beta/log/version are relational; graph fits in RAM.
6. **Event-sourced truth** (GIVEN structure + immutable outcome log; weights
   derived) — closes the persistence gap *and* preserves reproducibility/honesty.
7. **Schema evolution = stable skeleton + jsonb vocabulary + type registry +
   upcasters** — new types/keys cost no DDL and no core change; renames cost one
   upcaster.
8. **Ingestion = hybrid** (batch structure, event outcomes); one adapter
   interface, two hosts (propose vs decide).
9. **Graph grain = service/config-knob/datastore/incident**; code-parse collapses
   *up* to services; **only service-owning repos** are ingested (registered).
10. **Incident feed = trigger vs. measure split** (alert/SLO/manual trigger;
    blast set measured from traces); agent-action-as-outcome as a fast-follow.
11. **LLM = extract (ingest) + serve/explain (query) + offline baseline; never in
    the prediction path.**
12. **Service decomposition = 7 deployables, Python throughout, engine embedded in
    the read service, thin MCP adapter, sync read / async workers**; collapse is a
    deployment choice.
13. **Structural-learning governance = hybrid** (auto-low-confidence; human review
    above a surprise/impact threshold).
14. **Forgetting (decay + structure-change invalidation, dormant-not-deleted) =
    designed-in now, built in a later phase.**
15. **MVP = a walking skeleton** delivering the full loop on the real system with
    manual outcomes, one env, one process + CLI.

---

## 14. Open questions resolved & future directions

### The brief's four open questions — resolved
- **Graph store** → Postgres SoR + in-memory projection (§4); graph DB only if it
  ever outgrows memory.
- **Ingestion model** → hybrid: batch structure, event outcomes (§6).
- **Incident signals** → trigger (alert/SLO/manual) vs. measure (traces) split
  (§6, §8).
- **LLM placement** → extraction + narration + offline baseline; never the
  predictor (§7).

### Future (noted, not designed here)
- **Multi-tenant / SaaS** — the per-org boundaries (env tagging, access control,
  the engine/integration split) are SaaS-compatible, but multi-tenant specifics
  are deliberately out of scope.
- **Substrate swap** — the architecture (*signals → propose/decide → queryable
  model → outcome loop*) and metrics are substrate-agnostic; a learned/latent
  model could sit underneath without changing the contract.
- **Additional agents & deterministic proactivity** — hardened rule kits for
  Cursor/Cline/Windsurf and the Claude Code hook forcing-function (P4).
- **Richer outcome sources** — agent-action-as-outcome; CI/deploy signals;
  fault-injection campaigns.

---

*This is an internal design doc (under the gitignored `docs/superpowers/`), the
working artifact behind the productization of the OWM spike. The public-facing
architecture narrative lives in `README.md` and `docs/architecture-diagrams.md`.*
