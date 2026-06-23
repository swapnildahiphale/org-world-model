# OWM MVP Walking Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the smallest end-to-end OWM productization slice — persist the causal model in Postgres, ingest one+ service repo's structure, serve a calibrated *grounding pack* to a coding agent over MCP, and deepen the model from a manually-recorded incident — all in one process + a CLI, reusing the validated `owm/` core unchanged.

**Architecture:** A new `owm_server/` package wraps the proven `owm/` library (graph/propagate/learn) with: a Postgres system-of-record (event-sourced: GIVEN structure + append-only logs are durable, Beta weights are *derived*), an in-memory `CausalWorldModel` projection rebuilt from the DB for inference, a FastAPI Query/Grounding API, a thin MCP server (≈4 tools), structure-ingestion adapters (code-parse + topology), a learning step (surprise-gated `observe()`), and a CLI for batch ingest + manual outcomes. This is the roadmap's **MVP cut** — components ①MCP ②Read ③Postgres ④Ingestion ⑥Learning collapsed into one deployable; ⑤Outcome-receiver automation and ⑦ live-eval are later phases (here, outcomes are recorded manually).

**Tech Stack:** Python ≥3.10 · `owm/` core (networkx/numpy) · SQLAlchemy 2.x + Alembic · Postgres 16 (jsonb) · psycopg 3 · FastAPI + uvicorn · pydantic v2 · official `mcp` Python SDK · pytest + `testcontainers[postgres]`.

## Global Constraints

- **Scope:** MVP walking skeleton only. Roadmap phases P1–P5 (automated outcome receiver, continuous learning worker + hybrid governance, scale-out + UI, CC hooks + LLM extraction + multi-env + forgetting, honesty-at-scale) are **out of scope** and each get their own plan. Source of truth for design: `docs/superpowers/specs/2026-06-23-owm-architecture-design.md`.
- **Reuse, don't rewrite:** `owm/graph.py`, `owm/propagate.py`, `owm/learn.py`, `owm/codegraph.py`, `owm/topology.py` are used as-is. Do **not** modify `owm/` in this plan. Confirm exact constant names by reading them once: `Kind`/`Rel` and the `source` constants (`GIVEN`/`LEARNED`) in `owm/graph.py:26-58` and `owm/learn.py` (`LEARNED`, `NEW_EDGE_ALPHA`, `NEW_EDGE_BETA`, `SURPRISE_THRESHOLD`).
- **Event-sourced truth:** never hand-edit derived weights; the `incident_log` + GIVEN structure are the durable record. Learned (`HIDDEN`/`COUPLES`) edges are re-derivable by replaying the log through `observe()`.
- **One env:** every node/edge/log row carries `env`, defaulting to `"prod"`. Multi-env is a later phase — do not add env-conditioning logic now beyond storing the column.
- **Numbering:** circled refs ①–⑦ map to the spec's deployables (①MCP ②Read ③Postgres ④Ingestion ⑤Outcome ⑥Learning ⑦Eval).
- **Tests use a real Postgres** (jsonb + SQL is the point) via an ephemeral `testcontainers` Postgres — never SQLite, never mocks for the store layer.
- **Commit style:** Conventional Commits (`feat:`/`test:`/`chore:`). No AI attribution in commit messages.
- **Package import root:** `owm_server` (sibling to `owm`). All new code under `owm_server/`, all new tests under `tests/server/`.

---

## File Structure

```
owm_server/
  __init__.py
  config.py            # settings: DATABASE_URL, env default, surprise threshold
  db.py                # SQLAlchemy engine/session factory
  models.py            # ORM: Node, Edge, IncidentLog, PredictionLog, ModelVersion, OwnershipIndex
  store.py             # repository: upsert/query nodes+edges, append logs, version bump, ownership lookup
  projection.py        # build owm.graph.CausalWorldModel from store rows (reuse owm/)
  grounding.py         # GroundingPack (pydantic) + assemble(): localize → predict_blast → pack + persist prediction
  learn_service.py     # record_outcome(): predict → observe() → persist edges + incident + version bump
  ingest/
    __init__.py
    envelope.py        # ProposeEnvelope / OutcomeEnvelope (pydantic) — the adapter contract
    codeparse.py       # code-parse adapter: parse_repo → ownership index + config-knob nodes → store
    topology.py        # topology adapter: seed_graph → service nodes + GIVEN edges → store
  api.py               # FastAPI app: POST /ground, /blast_radius, /record_outcome
  mcp_server.py        # MCP server: owm_ground, owm_blast_radius, owm_why_coupled, owm_record_outcome
  cli.py               # CLI: owm ingest / owm ground / owm record-outcome
  integration_kit/
    AGENTS.md          # universal always-on rule (Codex/Cursor/Cline/Windsurf)
    cursor.owm.mdc     # Cursor always-apply rule
    cline.owm.md       # Cline rule
    windsurf.owm.md    # Windsurf always_on rule
migrations/            # Alembic
  env.py
  versions/0001_baseline.py
docker-compose.yml     # local Postgres 16 (dev only; tests use testcontainers)
tests/server/
  conftest.py          # ephemeral Postgres fixture + schema setup + session
  test_db.py
  test_models.py
  test_store.py
  test_projection.py
  test_grounding.py
  test_api.py
  test_ingest.py
  test_learn_service.py
  test_cli.py
  test_mcp_server.py
  test_integration_kit.py
```

---

### Task 1: Package scaffold, dependencies, and ephemeral-Postgres test harness

**Files:**
- Modify: `pyproject.toml` (add an `[project.optional-dependencies] server` + `server-dev` group)
- Create: `owm_server/__init__.py`, `owm_server/config.py`, `owm_server/db.py`
- Create: `docker-compose.yml`
- Test: `tests/server/conftest.py`, `tests/server/test_db.py`

**Interfaces:**
- Produces: `owm_server.config.Settings` (`.database_url: str`, `.default_env: str = "prod"`, `.surprise_threshold: float`); `owm_server.db.make_engine(url) -> Engine`, `owm_server.db.session_scope(engine) -> contextmanager[Session]`.
- Consumes: nothing (foundation task).

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

```toml
# under [project.optional-dependencies]
server = [
  "sqlalchemy>=2.0",
  "psycopg[binary]>=3.1",
  "alembic>=1.13",
  "fastapi>=0.110",
  "uvicorn>=0.29",
  "pydantic>=2.6",
  "pydantic-settings>=2.2",
  "mcp>=1.2",
  "typer>=0.12",
]
server-dev = [
  "pytest>=7",
  "httpx>=0.27",            # FastAPI TestClient
  "testcontainers[postgres]>=4.0",
]
```

- [ ] **Step 2: Write the failing test for config + db connectivity**

```python
# tests/server/conftest.py
import pytest
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer
from owm_server.db import make_engine, session_scope
from owm_server.models import Base

@pytest.fixture(scope="session")
def pg_url():
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg.get_connection_url().replace("psycopg2", "psycopg")

@pytest.fixture()
def engine(pg_url):
    eng = make_engine(pg_url)
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    return eng

@pytest.fixture()
def session(engine):
    with session_scope(engine) as s:
        yield s
```

```python
# tests/server/test_db.py
from sqlalchemy import text
from owm_server.db import session_scope

def test_connects_and_selects_one(engine):
    with session_scope(engine) as s:
        assert s.execute(text("SELECT 1")).scalar() == 1
```

- [ ] **Step 3: Run to verify it fails**

Run: `pip install -e ".[server,server-dev]" && pytest tests/server/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.db` (and `owm_server.models`).

- [ ] **Step 4: Implement `config.py` and `db.py`**

```python
# owm_server/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OWM_", env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://owm:owm@localhost:5432/owm"
    default_env: str = "prod"
    surprise_threshold: float = 0.10

settings = Settings()
```

```python
# owm_server/db.py
from contextlib import contextmanager
from collections.abc import Iterator
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)

@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    s = factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
```

(`owm_server/models.py` with `Base` is created in Task 2; the test imports it via conftest. If running Task 1 in isolation, create a temporary empty `Base = declarative_base()` placeholder and replace it in Task 2. Per execution order, do Task 2 next.)

- [ ] **Step 5: Create `docker-compose.yml` (dev convenience; tests don't use it)**

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment: { POSTGRES_USER: owm, POSTGRES_PASSWORD: owm, POSTGRES_DB: owm }
    ports: ["5432:5432"]
```

- [ ] **Step 6: Run to verify it passes**

Run: `pytest tests/server/test_db.py -v`
Expected: PASS (testcontainers pulls postgres:16-alpine on first run).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml owm_server/ docker-compose.yml tests/server/conftest.py tests/server/test_db.py
git commit -m "feat(server): scaffold owm_server package + ephemeral-postgres test harness"
```

---

### Task 2: Schema models + Alembic baseline

**Files:**
- Create: `owm_server/models.py`
- Create: `migrations/env.py`, `migrations/versions/0001_baseline.py`, `alembic.ini`
- Test: `tests/server/test_models.py`

**Interfaces:**
- Produces: `Base` (declarative) and ORM classes `Node, Edge, IncidentLog, PredictionLog, ModelVersion, OwnershipIndex` with the columns below. Later tasks import these.
- Consumes: `owm_server.db` (Task 1).

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_models.py
from owm_server.models import Node, Edge, ModelVersion
from owm_server.db import session_scope

def test_node_edge_roundtrip_with_jsonb(engine):
    with session_scope(engine) as s:
        s.add(Node(key="frontend", type="service", attrs={"lang": "go"}, env="prod"))
        s.add(Node(key="productcatalogservice::EXTRA_LATENCY", type="config_knob",
                   attrs={"default": "0s"}, env="prod"))
        s.flush()
        s.add(Edge(src="frontend", dst="productcatalogservice", rel="calls",
                   provenance="GIVEN", beta_alpha=1.0, beta_beta=1.0,
                   state_mask=["healthy", "saturated"], status="active", env="prod"))
    with session_scope(engine) as s:
        n = s.get(Node, "frontend")
        assert n.attrs["lang"] == "go"
        e = s.query(Edge).filter_by(src="frontend").one()
        assert e.provenance == "GIVEN" and e.beta_alpha == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'Node' from 'owm_server.models'`.

- [ ] **Step 3: Implement `models.py`**

```python
# owm_server/models.py
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

def _now() -> datetime:
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass

class Node(Base):
    __tablename__ = "node"
    key: Mapped[str] = mapped_column(String, primary_key=True)        # stable id, e.g. "frontend"
    type: Mapped[str] = mapped_column(String)                          # service|config_knob|datastore|incident
    attrs: Mapped[dict] = mapped_column(JSONB, default=dict)
    provenance: Mapped[str] = mapped_column(String, default="GIVEN")   # GIVEN|HIDDEN
    env: Mapped[str] = mapped_column(String, default="prod")
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

class Edge(Base):
    __tablename__ = "edge"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    src: Mapped[str] = mapped_column(String, index=True)
    dst: Mapped[str] = mapped_column(String, index=True)
    rel: Mapped[str] = mapped_column(String)                           # calls|depends_on|couples|config_affects
    provenance: Mapped[str] = mapped_column(String)                    # GIVEN|HIDDEN
    beta_alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta_beta: Mapped[float] = mapped_column(Float, default=1.0)
    state_mask: Mapped[list] = mapped_column(ARRAY(String), default=list)
    status: Mapped[str] = mapped_column(String, default="active")      # active|candidate|dormant
    saturation_sensitive: Mapped[bool] = mapped_column(default=False)
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)        # {incident_ids: [...]}
    model_version_introduced: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"), nullable=True)
    env: Mapped[str] = mapped_column(String, default="prod")

class IncidentLog(Base):
    __tablename__ = "incident_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    declared_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    trigger_src: Mapped[str] = mapped_column(String, default="manual")
    change_ref: Mapped[dict] = mapped_column(JSONB)                    # {origin_service|origin_config, ...}
    observed_blast_set: Mapped[list] = mapped_column(ARRAY(String))
    state_at_time: Mapped[str] = mapped_column(String, default="healthy")
    env: Mapped[str] = mapped_column(String, default="prod")
    raw_signals: Mapped[dict] = mapped_column(JSONB, default=dict)
    schema_ver: Mapped[int] = mapped_column(Integer, default=1)

class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    made_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    change_ref: Mapped[dict] = mapped_column(JSONB)
    predicted_blast: Mapped[dict] = mapped_column(JSONB)               # {service: prob}
    state: Mapped[str] = mapped_column(String, default="healthy")
    model_version: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"), nullable=True)
    grounding_pack: Mapped[dict] = mapped_column(JSONB, default=dict)
    schema_ver: Mapped[int] = mapped_column(Integer, default=1)

class ModelVersion(Base):
    __tablename__ = "model_version"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent: Mapped[int | None] = mapped_column(ForeignKey("model_version.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    rule_version: Mapped[str] = mapped_column(String, default="v1")
    summary: Mapped[str] = mapped_column(String, default="")
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)

class OwnershipIndex(Base):
    __tablename__ = "ownership_index"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo: Mapped[str] = mapped_column(String, index=True)
    path_glob: Mapped[str] = mapped_column(String)
    service: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String)                        # manifest|codeowners|convention|image
    conf: Mapped[float] = mapped_column(Float, default=1.0)
```

- [ ] **Step 4: Wire Alembic (baseline = autogenerate from metadata)**

```ini
; alembic.ini (minimal)
[alembic]
script_location = migrations
sqlalchemy.url = postgresql+psycopg://owm:owm@localhost:5432/owm
```

```python
# migrations/env.py (offline+online, target_metadata = Base.metadata)
from alembic import context
from sqlalchemy import engine_from_config, pool
from owm_server.models import Base
config = context.config
target_metadata = Base.metadata
def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section),
                                     prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
run_migrations_online()
```

Generate the baseline against a running dev Postgres: `alembic revision --autogenerate -m baseline` → commit the produced `migrations/versions/0001_baseline.py`. (Tests create the schema via `Base.metadata.create_all` in the conftest fixture, so they don't depend on Alembic; Alembic is for real deployments.)

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/server/test_models.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add owm_server/models.py migrations/ alembic.ini tests/server/test_models.py
git commit -m "feat(server): event-sourced schema (node/edge/logs/version/ownership) + alembic baseline"
```

---

### Task 3: Store / repository layer

**Files:**
- Create: `owm_server/store.py`
- Test: `tests/server/test_store.py`

**Interfaces:**
- Produces:
  - `Store(session)` with: `upsert_node(key, type, attrs=None, provenance="GIVEN", env="prod")`,
    `upsert_edge(src, dst, rel, provenance, alpha=1.0, beta=1.0, state_mask=None, status="active", saturation_sensitive=False, env="prod") -> int`,
    `add_ownership(repo, path_glob, service, source, conf=1.0)`,
    `service_for_path(path) -> str | None`,
    `append_incident(change_ref, observed_blast_set, state_at_time, env="prod", trigger_src="manual") -> int`,
    `append_prediction(change_ref, predicted_blast, state, model_version, grounding_pack) -> int`,
    `new_version(parent=None, summary="", metrics=None) -> int`,
    `current_version() -> int | None`,
    `all_nodes() -> list[Node]`, `all_edges() -> list[Edge]`.
- Consumes: `owm_server.models`, `owm_server.db` (Tasks 1–2).

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_store.py
from owm_server.store import Store

def test_upsert_is_idempotent_and_versions_bump(session):
    st = Store(session)
    st.upsert_node("frontend", "service")
    st.upsert_node("frontend", "service", attrs={"lang": "go"})   # second upsert updates, not duplicates
    eid = st.upsert_edge("frontend", "cartservice", "calls", "GIVEN")
    assert isinstance(eid, int)
    assert len(st.all_nodes()) == 1
    v1 = st.new_version(summary="ingest")
    v2 = st.new_version(parent=v1, summary="learn")
    assert v2 != v1 and st.current_version() == v2

def test_ownership_lookup(session):
    st = Store(session)
    st.add_ownership("repo-a", "src/frontend/**", "frontend", "convention")
    assert st.service_for_path("src/frontend/main.go") == "frontend"
    assert st.service_for_path("src/unknown/x.go") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.store`.

- [ ] **Step 3: Implement `store.py`**

```python
# owm_server/store.py
import fnmatch
from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from owm_server.models import Node, Edge, IncidentLog, PredictionLog, ModelVersion, OwnershipIndex

class Store:
    def __init__(self, session: Session):
        self.s = session

    def upsert_node(self, key, type, attrs=None, provenance="GIVEN", env="prod"):
        n = self.s.get(Node, key)
        if n is None:
            self.s.add(Node(key=key, type=type, attrs=attrs or {}, provenance=provenance, env=env))
        else:
            n.type = type
            if attrs:
                n.attrs = {**n.attrs, **attrs}
        self.s.flush()

    def upsert_edge(self, src, dst, rel, provenance, alpha=1.0, beta=1.0,
                    state_mask=None, status="active", saturation_sensitive=False, env="prod") -> int:
        e = self.s.query(Edge).filter_by(src=src, dst=dst, rel=rel, env=env).one_or_none()
        if e is None:
            e = Edge(src=src, dst=dst, rel=rel, provenance=provenance,
                     beta_alpha=alpha, beta_beta=beta, state_mask=state_mask or [],
                     status=status, saturation_sensitive=saturation_sensitive, env=env)
            self.s.add(e)
        else:
            e.provenance, e.beta_alpha, e.beta_beta = provenance, alpha, beta
            e.status, e.saturation_sensitive = status, saturation_sensitive
            if state_mask is not None:
                e.state_mask = state_mask
        self.s.flush()
        return e.id

    def add_ownership(self, repo, path_glob, service, source, conf=1.0):
        self.s.add(OwnershipIndex(repo=repo, path_glob=path_glob, service=service, source=source, conf=conf))
        self.s.flush()

    def service_for_path(self, path) -> str | None:
        rows = self.s.execute(select(OwnershipIndex).order_by(desc(OwnershipIndex.conf))).scalars().all()
        for r in rows:
            if fnmatch.fnmatch(path, r.path_glob):
                return r.service
        return None

    def append_incident(self, change_ref, observed_blast_set, state_at_time, env="prod", trigger_src="manual") -> int:
        row = IncidentLog(change_ref=change_ref, observed_blast_set=list(observed_blast_set),
                          state_at_time=state_at_time, env=env, trigger_src=trigger_src)
        self.s.add(row); self.s.flush()
        return row.id

    def append_prediction(self, change_ref, predicted_blast, state, model_version, grounding_pack) -> int:
        row = PredictionLog(change_ref=change_ref, predicted_blast=predicted_blast, state=state,
                            model_version=model_version, grounding_pack=grounding_pack)
        self.s.add(row); self.s.flush()
        return row.id

    def new_version(self, parent=None, summary="", metrics=None) -> int:
        mv = ModelVersion(parent=parent, summary=summary, metrics=metrics or {})
        self.s.add(mv); self.s.flush()
        return mv.id

    def current_version(self) -> int | None:
        return self.s.execute(select(ModelVersion.id).order_by(desc(ModelVersion.id)).limit(1)).scalar()

    def all_nodes(self) -> list[Node]:
        return self.s.execute(select(Node)).scalars().all()

    def all_edges(self) -> list[Edge]:
        return self.s.execute(select(Edge)).scalars().all()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add owm_server/store.py tests/server/test_store.py
git commit -m "feat(server): store/repository layer over the event-sourced schema"
```

---

### Task 4: In-memory projection (reuse `owm.graph`)

**Files:**
- Create: `owm_server/projection.py`
- Test: `tests/server/test_projection.py`

**Interfaces:**
- Produces: `build_projection(store: Store) -> CausalWorldModel`. Reads all nodes/edges, reconstructs a `CausalWorldModel` (from `owm.graph`) with `weight = beta_weight(alpha, beta)` and the original `source`/`saturation_sensitive`.
- Consumes: `owm_server.store.Store`; `owm.graph.CausalWorldModel`, `owm.topology.beta_weight`. Read `owm/graph.py:26-58` for the exact `Kind`/`Rel` and `source` constant strings; map DB `type`→`Kind.*`, DB `rel`→`Rel.*`, DB `provenance` GIVEN→`source="given"`/HIDDEN→`LEARNED` (use the literal strings the core expects).

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_projection.py
from owm_server.store import Store
from owm_server.projection import build_projection

def test_projection_reconstructs_nodes_and_edges(session):
    st = Store(session)
    st.upsert_node("frontend", "service")
    st.upsert_node("cartservice", "service")
    st.upsert_edge("frontend", "cartservice", "calls", "GIVEN", alpha=1.0, beta=1.0)
    st.upsert_edge("pcs::EXTRA_LATENCY", "frontend", "couples", "HIDDEN",
                   alpha=4.0, beta=1.0, saturation_sensitive=True)
    cwm = build_projection(st)
    assert "frontend" in cwm.services
    assert cwm.g.has_edge("frontend", "cartservice")
    assert cwm.g.has_edge("pcs::EXTRA_LATENCY", "frontend")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_projection.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.projection`.

- [ ] **Step 3: Implement `projection.py`**

```python
# owm_server/projection.py
from owm.graph import CausalWorldModel, Kind, Rel
from owm.topology import beta_weight
from owm_server.store import Store

_TYPE_TO_KIND = {"service": Kind.SERVICE, "config_knob": Kind.CONFIG,
                 "datastore": Kind.DATASTORE, "incident": Kind.INCIDENT}
# DB rel string -> core Rel constant
_REL = {"calls": Rel.CALLS, "depends_on": Rel.DEPENDS_ON, "couples": Rel.COUPLES,
        "config_affects": Rel.CONFIG_AFFECTS}

def build_projection(store: Store) -> CausalWorldModel:
    cwm = CausalWorldModel()
    for n in store.all_nodes():
        cwm.add_node(n.key, _TYPE_TO_KIND.get(n.type, Kind.SERVICE), **(n.attrs or {}))
    for e in store.all_edges():
        if e.status == "dormant":
            continue
        cwm.add_edge(
            e.src, e.dst, _REL.get(e.rel, e.rel),
            weight=beta_weight(e.beta_alpha, e.beta_beta),
            source=("given" if e.provenance == "GIVEN" else "learned"),
            saturation_sensitive=e.saturation_sensitive,
            alpha=e.beta_alpha, beta=e.beta_beta,
        )
    return cwm
```

> Note: confirm the exact member names on `Kind`/`Rel` (e.g. `Kind.SERVICE`, `Rel.CALLS`, `Rel.DEPENDS_ON`, `Rel.COUPLES`, `Rel.CONFIG_AFFECTS`) and the `source` literals the core stores by reading `owm/graph.py`; adjust the two maps if the constants differ. If `add_node` rejects unknown kwargs, filter `attrs` to the keys the core accepts.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_projection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add owm_server/projection.py tests/server/test_projection.py
git commit -m "feat(server): build in-memory CausalWorldModel projection from the store"
```

---

### Task 5: Grounding assembly (reuse `owm.propagate`)

**Files:**
- Create: `owm_server/grounding.py`
- Test: `tests/server/test_grounding.py`

**Interfaces:**
- Produces:
  - pydantic `ChangeRef(service: str | None = None, config: str | None = None, path: str | None = None)`,
    `BlastItem(service: str, p: float)`, `GroundingPack(blast_radius: list[BlastItem], hidden_couplings: list[dict], ownership: list[dict], related_incidents: list[dict], abstain: dict, model_version: int | None)`.
  - `localize(store, change: ChangeRef) -> tuple[list[str], list[str]]` → `(touched_services, touched_configs)`.
  - `assemble(store, change: ChangeRef, state: str = "healthy", persist: bool = True) -> GroundingPack`.
- Consumes: `owm_server.store.Store`, `owm_server.projection.build_projection`, `owm.propagate.predict_blast`.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_grounding.py
from owm_server.store import Store
from owm_server.grounding import ChangeRef, assemble

def test_grounding_ranks_learned_coupling(session):
    st = Store(session)
    for svc in ["frontend", "recommendationservice", "cartservice", "productcatalogservice"]:
        st.upsert_node(svc, "service")
    st.upsert_node("productcatalogservice::EXTRA_LATENCY", "config_knob")
    # learned hidden coupling: the config knob couples to frontend + reco
    st.upsert_edge("productcatalogservice::EXTRA_LATENCY", "frontend", "couples", "HIDDEN", alpha=8, beta=1, saturation_sensitive=True)
    st.upsert_edge("productcatalogservice::EXTRA_LATENCY", "recommendationservice", "couples", "HIDDEN", alpha=8, beta=1, saturation_sensitive=True)
    st.new_version(summary="seed")
    pack = assemble(st, ChangeRef(config="productcatalogservice::EXTRA_LATENCY"), state="saturated")
    ranked = {b.service: b.p for b in pack.blast_radius}
    assert ranked["frontend"] > 0.5
    assert ranked["recommendationservice"] > 0.5
    assert ranked.get("cartservice", 0.0) < ranked["frontend"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_grounding.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.grounding`.

- [ ] **Step 3: Implement `grounding.py`**

```python
# owm_server/grounding.py
from pydantic import BaseModel
from owm.propagate import predict_blast
from owm_server.store import Store
from owm_server.projection import build_projection

class ChangeRef(BaseModel):
    service: str | None = None
    config: str | None = None
    path: str | None = None

class BlastItem(BaseModel):
    service: str
    p: float

class GroundingPack(BaseModel):
    blast_radius: list[BlastItem]
    hidden_couplings: list[dict]
    ownership: list[dict]
    related_incidents: list[dict]
    abstain: dict
    model_version: int | None

def localize(store: Store, change: ChangeRef) -> tuple[list[str], list[str]]:
    services, configs = [], []
    if change.config:
        configs.append(change.config)
    if change.service:
        services.append(change.service)
    if change.path:
        svc = store.service_for_path(change.path)
        if svc:
            services.append(svc)
    return services, configs

def assemble(store: Store, change: ChangeRef, state: str = "healthy", persist: bool = True) -> GroundingPack:
    cwm = build_projection(store)
    touched_services, touched_configs = localize(store, change)
    probs = predict_blast(cwm, touched_services=touched_services or None,
                          touched_configs=touched_configs or None, s_t=state)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    blast = [BlastItem(service=s, p=round(p, 4)) for s, p in ranked if p > 0.0]

    # hidden couplings touching the origin (learned edges only)
    origins = set(touched_services) | set(touched_configs)
    hidden = [{"a": e.src, "b": e.dst, "alpha": e.beta_alpha, "beta": e.beta_beta}
              for e in store.all_edges()
              if e.provenance == "HIDDEN" and (e.src in origins or e.dst in origins)]

    top = blast[0].p if blast else 0.0
    abstain = {"should_ask": top < 0.2, "reason": "low confidence" if top < 0.2 else ""}
    mv = store.current_version()
    pack = GroundingPack(blast_radius=blast, hidden_couplings=hidden, ownership=[],
                         related_incidents=[], abstain=abstain, model_version=mv)
    if persist:
        store.append_prediction(change.model_dump(), {b.service: b.p for b in blast},
                                state, mv, pack.model_dump())
    return pack
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_grounding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add owm_server/grounding.py tests/server/test_grounding.py
git commit -m "feat(server): grounding pack assembly (localize + predict_blast + persist prediction)"
```

---

### Task 6: Query/Grounding API (FastAPI)

**Files:**
- Create: `owm_server/api.py`
- Test: `tests/server/test_api.py`

**Interfaces:**
- Produces: `create_app(engine) -> FastAPI` with `POST /ground` (body `{change, state}` → `GroundingPack`), `POST /blast_radius` (→ `{blast_radius}`), `POST /record_outcome` (body `{change, observed_blast_set, state}` → `{model_version}`). `/record_outcome` delegates to Task 8's `record_outcome` (import lazily; until Task 8 exists, this endpoint test is written in Task 8).
- Consumes: `owm_server.grounding`, `owm_server.db`.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_api.py
from fastapi.testclient import TestClient
from owm_server.api import create_app
from owm_server.store import Store

def test_ground_endpoint_returns_pack(engine):
    with __import__("owm_server.db", fromlist=["session_scope"]).session_scope(engine) as s:
        st = Store(s)
        st.upsert_node("frontend", "service")
        st.upsert_node("pcs::KNOB", "config_knob")
        st.upsert_edge("pcs::KNOB", "frontend", "couples", "HIDDEN", alpha=8, beta=1, saturation_sensitive=True)
        st.new_version(summary="seed")
    client = TestClient(create_app(engine))
    r = client.post("/ground", json={"change": {"config": "pcs::KNOB"}, "state": "saturated"})
    assert r.status_code == 200
    body = r.json()
    assert any(b["service"] == "frontend" and b["p"] > 0.5 for b in body["blast_radius"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.api`.

- [ ] **Step 3: Implement `api.py`**

```python
# owm_server/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from owm_server.db import session_scope
from owm_server.store import Store
from owm_server.grounding import ChangeRef, assemble

class GroundReq(BaseModel):
    change: ChangeRef
    state: str = "healthy"

class OutcomeReq(BaseModel):
    change: ChangeRef
    observed_blast_set: list[str]
    state: str = "healthy"

def create_app(engine) -> FastAPI:
    app = FastAPI(title="OWM Grounding API")

    @app.post("/ground")
    def ground(req: GroundReq):
        with session_scope(engine) as s:
            return assemble(Store(s), req.change, req.state).model_dump()

    @app.post("/blast_radius")
    def blast_radius(req: GroundReq):
        with session_scope(engine) as s:
            pack = assemble(Store(s), req.change, req.state, persist=False)
            return {"blast_radius": [b.model_dump() for b in pack.blast_radius]}

    @app.post("/record_outcome")
    def record_outcome_ep(req: OutcomeReq):
        from owm_server.learn_service import record_outcome   # Task 8
        with session_scope(engine) as s:
            mv = record_outcome(Store(s), req.change, req.observed_blast_set, req.state)
            return {"model_version": mv}

    return app
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_api.py::test_ground_endpoint_returns_pack -v`
Expected: PASS (the `/record_outcome` path is exercised in Task 8).

- [ ] **Step 5: Commit**

```bash
git add owm_server/api.py tests/server/test_api.py
git commit -m "feat(server): FastAPI grounding API (/ground, /blast_radius, /record_outcome)"
```

---

### Task 7: Structure ingestion adapters (code-parse + topology)

**Files:**
- Create: `owm_server/ingest/__init__.py`, `owm_server/ingest/envelope.py`, `owm_server/ingest/codeparse.py`, `owm_server/ingest/topology.py`
- Test: `tests/server/test_ingest.py`

**Interfaces:**
- Produces:
  - `envelope.ProposeEnvelope(kind: str, nodes: list[dict], edges: list[dict], ownership: list[dict])`.
  - `codeparse.build_envelope(repo_root: Path, repo_name: str) -> ProposeEnvelope` (reuses `owm.codegraph.parse_repo`; emits service + config-knob nodes + ownership rows).
  - `topology.build_envelope(repo_root: Path) -> ProposeEnvelope` (reuses `owm.codegraph.parse_repo` + `owm.topology.seed_graph`; emits GIVEN service→service edges from the seeded graph).
  - `apply_envelope(store: Store, env: ProposeEnvelope, repo_name: str)` — upserts nodes/edges/ownership; bumps a version with summary `"ingest:<repo>"`.
- Consumes: `owm.codegraph.parse_repo`, `owm.topology.seed_graph`, `owm_server.store.Store`. Uses fixture repo `tests/fixtures/ob_mini`.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_ingest.py
from pathlib import Path
from owm_server.store import Store
from owm_server.ingest import codeparse, topology, apply_envelope

OB = Path("tests/fixtures/ob_mini")

def test_codeparse_emits_services_and_ownership(session):
    st = Store(session)
    env = codeparse.build_envelope(OB, repo_name="ob")
    apply_envelope(st, env, "ob")
    svcs = {n.key for n in st.all_nodes() if n.type == "service"}
    assert "frontend" in svcs and "productcatalogservice" in svcs
    assert st.service_for_path("src/frontend/main.go") == "frontend"

def test_topology_emits_given_call_edges(session):
    st = Store(session)
    apply_envelope(st, codeparse.build_envelope(OB, "ob"), "ob")
    apply_envelope(st, topology.build_envelope(OB), "ob")
    given = [(e.src, e.dst) for e in st.all_edges() if e.provenance == "GIVEN" and e.rel == "calls"]
    assert any(src == "frontend" for src, _ in given)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.ingest`.

- [ ] **Step 3: Implement the ingest package**

```python
# owm_server/ingest/envelope.py
from pydantic import BaseModel
class ProposeEnvelope(BaseModel):
    kind: str
    nodes: list[dict] = []
    edges: list[dict] = []
    ownership: list[dict] = []
```

```python
# owm_server/ingest/codeparse.py
from pathlib import Path
from owm.codegraph import parse_repo
from owm_server.ingest.envelope import ProposeEnvelope

def build_envelope(repo_root: Path, repo_name: str) -> ProposeEnvelope:
    pc = parse_repo(repo_root)
    nodes, ownership = [], []
    # services + per-service config knobs from ParsedCode
    # NOTE: read owm/codegraph.py:25-110 for the exact ParsedCode attribute names;
    # the names below (services, file_to_service, config_knobs) may differ — adjust.
    for svc in pc.services:
        nodes.append({"key": svc, "type": "service"})
    for path, svc in pc.file_to_service.items():
        ownership.append({"repo": repo_name, "path_glob": path, "service": svc, "source": "manifest"})
    for svc, knobs in getattr(pc, "config_knobs", {}).items():
        for k in knobs:
            nodes.append({"key": f"{svc}::{k}", "type": "config_knob", "attrs": {"service": svc}})
    return ProposeEnvelope(kind="structure", nodes=nodes, ownership=ownership)
```

```python
# owm_server/ingest/topology.py
from pathlib import Path
from owm.codegraph import parse_repo
from owm.topology import seed_graph
from owm_server.ingest.envelope import ProposeEnvelope

def build_envelope(repo_root: Path) -> ProposeEnvelope:
    cwm = seed_graph(parse_repo(repo_root))
    edges = []
    for src, dst, data in cwm.edges_view():
        rel = data.get("rel") or data.get("key") or "calls"   # adjust to core's edge-data shape
        if rel == "calls":
            edges.append({"src": src, "dst": dst, "rel": "calls",
                          "alpha": data.get("alpha", 1.0), "beta": data.get("beta", 1.0)})
    return ProposeEnvelope(kind="structure", edges=edges)
```

```python
# owm_server/ingest/__init__.py
from owm_server.store import Store
from owm_server.ingest.envelope import ProposeEnvelope

def apply_envelope(store: Store, env: ProposeEnvelope, repo_name: str):
    for n in env.nodes:
        store.upsert_node(n["key"], n["type"], n.get("attrs"), provenance="GIVEN")
    for e in env.edges:
        store.upsert_edge(e["src"], e["dst"], e["rel"], "GIVEN",
                          alpha=e.get("alpha", 1.0), beta=e.get("beta", 1.0),
                          state_mask=["healthy", "saturated"])
    for o in env.ownership:
        store.add_ownership(o["repo"], o["path_glob"], o["service"], o["source"])
    store.new_version(summary=f"ingest:{repo_name}")
```

> Note: `ParsedCode`'s real attribute names and `CausalWorldModel.edges_view()`'s data dict are defined in `owm/codegraph.py` and `owm/graph.py` — read them once (Task 4 already inspected `edges_view`) and adjust the three `getattr`/`data.get` accesses so the fields match. The fixture `tests/fixtures/ob_mini` already exercises `parse_repo`/`seed_graph` in the existing suite, so use the existing tests as a reference for the real shapes.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_ingest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add owm_server/ingest/ tests/server/test_ingest.py
git commit -m "feat(server): structure ingestion adapters (code-parse + topology) over the store"
```

---

### Task 8: Learning step (reuse `owm.learn.observe`) — the deepening

**Files:**
- Create: `owm_server/learn_service.py`
- Test: `tests/server/test_learn_service.py`

**Interfaces:**
- Produces: `record_outcome(store: Store, change: ChangeRef, observed_blast_set: list[str], state: str = "healthy", learning_enabled: bool = True) -> int` — returns the new `model_version` id. It: builds the projection, computes `predicted_probs` via `predict_blast` (for surprise gating + the incident's pre-state), calls `observe(...)`, **persists** the resulting COUPLES edges back to the store (provenance `HIDDEN`, with the post-update `alpha`/`beta`), appends an `incident_log` row, and bumps the version.
- Consumes: `owm.learn.observe`, `owm.propagate.predict_blast`, `owm_server.projection`, `owm_server.grounding.ChangeRef`, `owm_server.store.Store`.

- [ ] **Step 1: Write the failing test (deepening + ablation)**

```python
# tests/server/test_learn_service.py
from owm_server.store import Store
from owm_server.grounding import ChangeRef, assemble
from owm_server.learn_service import record_outcome

def _seed(st):
    for svc in ["frontend", "recommendationservice", "checkoutservice", "productcatalogservice"]:
        st.upsert_node(svc, "service")
    st.upsert_node("productcatalogservice::EXTRA_LATENCY", "config_knob")
    st.new_version(summary="seed")

def test_outcome_deepens_then_grounding_predicts_coupling(session):
    st = Store(session)
    _seed(st)
    change = ChangeRef(config="productcatalogservice::EXTRA_LATENCY")
    before = {b.service: b.p for b in assemble(st, change, "saturated", persist=False).blast_radius}
    # record the measured blast set (the hidden coupling)
    record_outcome(st, change, ["frontend", "recommendationservice"], "saturated")
    after = {b.service: b.p for b in assemble(st, change, "saturated", persist=False).blast_radius}
    assert after.get("frontend", 0) > before.get("frontend", 0)
    assert after["frontend"] > 0.5

def test_ablation_learning_disabled_does_not_deepen(session):
    st = Store(session)
    _seed(st)
    change = ChangeRef(config="productcatalogservice::EXTRA_LATENCY")
    record_outcome(st, change, ["frontend"], "saturated", learning_enabled=False)
    after = {b.service: b.p for b in assemble(st, change, "saturated", persist=False).blast_radius}
    assert after.get("frontend", 0.0) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_learn_service.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.learn_service`.

- [ ] **Step 3: Implement `learn_service.py`**

```python
# owm_server/learn_service.py
from owm.propagate import predict_blast
from owm.learn import observe
from owm.graph import Rel
from owm_server.projection import build_projection
from owm_server.grounding import ChangeRef, localize
from owm_server.store import Store

def record_outcome(store: Store, change: ChangeRef, observed_blast_set, state="healthy",
                   learning_enabled: bool = True) -> int:
    cwm = build_projection(store)
    touched_services, touched_configs = localize(store, change)
    predicted = predict_blast(cwm, touched_services=touched_services or None,
                              touched_configs=touched_configs or None, s_t=state)
    # origin: the perturbed config or service node (observe() expects a single class node)
    origin = (touched_configs or touched_services)[0]
    observe(cwm, origin, observed_blast_set, predicted, learning_enabled=learning_enabled)

    # persist learned COUPLES edges (HIDDEN) with their post-update Beta params
    if learning_enabled:
        for src, dst, data in cwm.edges_view():
            if (data.get("rel") or data.get("key")) == Rel.COUPLES:
                store.upsert_edge(src, dst, "couples", "HIDDEN",
                                  alpha=data.get("alpha", 1.0), beta=data.get("beta", 1.0),
                                  state_mask=["saturated"], status="active",
                                  saturation_sensitive=bool(data.get("saturation_sensitive", True)))
    store.append_incident(change.model_dump(), observed_blast_set, state)
    return store.new_version(parent=store.current_version(),
                             summary=f"learn:{origin}",
                             metrics={"observed": list(observed_blast_set)})
```

> Note: align `data.get("rel")`/`data.get("alpha")` with the actual edge-data keys `CausalWorldModel.add_edge` stores (inspect via the Task 4 work). `observe()`'s `origin` must be the class node id that exists in the graph; the config-knob node key (`"<svc>::<KNOB>"`) is the right origin for a config change.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_learn_service.py -v && pytest tests/server/test_api.py -v`
Expected: PASS (both the deepening tests and the previously-deferred `/record_outcome` API path).

- [ ] **Step 5: Commit**

```bash
git add owm_server/learn_service.py tests/server/test_learn_service.py
git commit -m "feat(server): learning step — surprise-gated observe() persisted as the deepening loop"
```

---

### Task 9: CLI (batch ingest + manual outcome + ground)

**Files:**
- Create: `owm_server/cli.py`
- Modify: `pyproject.toml` (`[project.scripts] owm = "owm_server.cli:app"`)
- Test: `tests/server/test_cli.py`

**Interfaces:**
- Produces a Typer app `app` with commands: `ingest <repo_root> --name <repo>` (code-parse + topology), `ground --service/--config/--path [--state]` (prints the pack as JSON), `record-outcome --config/--service --impacted a,b,c [--state]`.
- Consumes: Tasks 5,7,8 + `owm_server.db`/`config`.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_cli.py
from typer.testing import CliRunner
from owm_server.cli import app, _engine_for_tests
import owm_server.cli as climod

def test_cli_ingest_then_ground(engine, monkeypatch):
    monkeypatch.setattr(climod, "get_engine", lambda: engine)
    r = CliRunner().invoke(app, ["ingest", "tests/fixtures/ob_mini", "--name", "ob"])
    assert r.exit_code == 0
    r2 = CliRunner().invoke(app, ["ground", "--service", "productcatalogservice", "--state", "healthy"])
    assert r2.exit_code == 0 and "blast_radius" in r2.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.cli`.

- [ ] **Step 3: Implement `cli.py`**

```python
# owm_server/cli.py
import json
from pathlib import Path
import typer
from owm_server.config import settings
from owm_server.db import make_engine, session_scope
from owm_server.models import Base
from owm_server.store import Store
from owm_server.grounding import ChangeRef, assemble
from owm_server.learn_service import record_outcome
from owm_server.ingest import codeparse, topology, apply_envelope

app = typer.Typer(help="OWM MVP CLI")

def get_engine():
    eng = make_engine(settings.database_url)
    Base.metadata.create_all(eng)
    return eng

def _engine_for_tests():  # referenced by tests; real runs use get_engine()
    return get_engine()

@app.command()
def ingest(repo_root: str, name: str = typer.Option(...)):
    eng = get_engine()
    with session_scope(eng) as s:
        st = Store(s)
        apply_envelope(st, codeparse.build_envelope(Path(repo_root), name), name)
        apply_envelope(st, topology.build_envelope(Path(repo_root)), name)
    typer.echo(f"ingested {name}")

@app.command()
def ground(service: str = typer.Option(None), config: str = typer.Option(None),
           path: str = typer.Option(None), state: str = typer.Option("healthy")):
    eng = get_engine()
    with session_scope(eng) as s:
        pack = assemble(Store(s), ChangeRef(service=service, config=config, path=path), state)
    typer.echo(json.dumps(pack.model_dump(), indent=2))

@app.command("record-outcome")
def record_outcome_cmd(impacted: str = typer.Option(...), service: str = typer.Option(None),
                       config: str = typer.Option(None), state: str = typer.Option("healthy")):
    eng = get_engine()
    with session_scope(eng) as s:
        mv = record_outcome(Store(s), ChangeRef(service=service, config=config),
                            [x.strip() for x in impacted.split(",")], state)
    typer.echo(f"deepened → model_version {mv}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add owm_server/cli.py pyproject.toml tests/server/test_cli.py
git commit -m "feat(server): CLI for batch ingest, ground, and manual outcome recording"
```

---

### Task 10: MCP server (thin, ≈4 tools)

**Files:**
- Create: `owm_server/mcp_server.py`
- Test: `tests/server/test_mcp_server.py`

**Interfaces:**
- Produces: a module-level `mcp` server (official `mcp` SDK `FastMCP`) registering tools `owm_ground`, `owm_blast_radius`, `owm_why_coupled`, `owm_record_outcome`, each with **trigger-rich docstrings** ("Call BEFORE editing code / changing config / infra…"). Each tool opens a `session_scope` and delegates to Tasks 5/8. Expose `owm_ground.fn` (the underlying function) for direct unit testing.
- Consumes: `owm_server.grounding`, `owm_server.learn_service`, `owm_server.db`.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_mcp_server.py
import owm_server.mcp_server as m
from owm_server.store import Store
from owm_server.db import session_scope

def test_owm_ground_tool_returns_pack(engine, monkeypatch):
    monkeypatch.setattr(m, "get_engine", lambda: engine)
    with session_scope(engine) as s:
        st = Store(s); st.upsert_node("frontend", "service")
        st.upsert_node("k::KNOB", "config_knob")
        st.upsert_edge("k::KNOB", "frontend", "couples", "HIDDEN", alpha=8, beta=1, saturation_sensitive=True)
        st.new_version(summary="seed")
    out = m.owm_ground(change={"config": "k::KNOB"}, state="saturated")
    assert any(b["service"] == "frontend" and b["p"] > 0.5 for b in out["blast_radius"])

def test_tools_are_registered():
    names = {t.name for t in m.mcp._tool_manager.list_tools()}
    assert {"owm_ground", "owm_blast_radius", "owm_why_coupled", "owm_record_outcome"} <= names
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError: owm_server.mcp_server`.

- [ ] **Step 3: Implement `mcp_server.py`**

```python
# owm_server/mcp_server.py
from mcp.server.fastmcp import FastMCP
from owm_server.config import settings
from owm_server.db import make_engine, session_scope
from owm_server.models import Base
from owm_server.store import Store
from owm_server.grounding import ChangeRef, assemble
from owm_server.learn_service import record_outcome

mcp = FastMCP("owm")

def get_engine():
    eng = make_engine(settings.database_url)
    Base.metadata.create_all(eng)
    return eng

@mcp.tool()
def owm_ground(change: dict, state: str = "healthy") -> dict:
    """Call BEFORE editing code, changing a config/schema, or modifying infra.
    Returns the org-specific predicted blast radius (calibrated), hidden couplings,
    and an abstain flag. `change` = {service|config|path}."""
    with session_scope(get_engine()) as s:
        return assemble(Store(s), ChangeRef(**change), state).model_dump()

@mcp.tool()
def owm_blast_radius(change: dict, state: str = "healthy") -> dict:
    """Ranked, calibrated blast radius only (no full pack). Call before acting on a change."""
    with session_scope(get_engine()) as s:
        pack = assemble(Store(s), ChangeRef(**change), state, persist=False)
        return {"blast_radius": [b.model_dump() for b in pack.blast_radius]}

@mcp.tool()
def owm_why_coupled(a: str, b: str) -> dict:
    """Explain the learned coupling between two services/configs (evidence + Beta confidence)."""
    with session_scope(get_engine()) as s:
        st = Store(s)
        edges = [{"src": e.src, "dst": e.dst, "provenance": e.provenance,
                  "alpha": e.beta_alpha, "beta": e.beta_beta, "evidence": e.evidence}
                 for e in st.all_edges()
                 if {e.src, e.dst} == {a, b}]
        return {"couplings": edges}

@mcp.tool()
def owm_record_outcome(change: dict, observed_blast_set: list[str], state: str = "healthy") -> dict:
    """Record what actually broke after a change so the model deepens. `change`={service|config}."""
    with session_scope(get_engine()) as s:
        mv = record_outcome(Store(s), ChangeRef(**change), observed_blast_set, state)
        return {"model_version": mv}

if __name__ == "__main__":
    mcp.run()   # stdio transport
```

> Note: the `FastMCP` tool-introspection attribute (`_tool_manager.list_tools()`) and `.fn` accessor depend on the installed `mcp` version; if the test's introspection path differs, adjust to the version's public API (e.g. `await mcp.list_tools()`). The functions are plain callables decorated by `@mcp.tool()`, so calling `owm_ground(...)` directly in the test works regardless.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_mcp_server.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add owm_server/mcp_server.py tests/server/test_mcp_server.py
git commit -m "feat(server): thin MCP server exposing owm_ground/blast_radius/why_coupled/record_outcome"
```

---

### Task 11: Integration kit (portable always-on rules)

**Files:**
- Create: `owm_server/integration_kit/AGENTS.md`, `owm_server/integration_kit/cursor.owm.mdc`, `owm_server/integration_kit/cline.owm.md`, `owm_server/integration_kit/windsurf.owm.md`
- Test: `tests/server/test_integration_kit.py`

**Interfaces:**
- Produces: shipped rule artifacts that steer any agent to call `owm_ground`/`owm_blast_radius` before edits/infra changes. No code; these are the portable proactivity layer (hooks are a later phase).
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/server/test_integration_kit.py
from pathlib import Path
KIT = Path("owm_server/integration_kit")
def test_kit_files_present_and_instructive():
    for f in ["AGENTS.md", "cursor.owm.mdc", "cline.owm.md", "windsurf.owm.md"]:
        p = KIT / f
        assert p.exists(), f"missing {f}"
        text = p.read_text().lower()
        assert "owm_ground" in text and "before" in text
    # cursor rule must be always-applied
    assert "alwaysapply: true" in (KIT / "cursor.owm.mdc").read_text().lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/server/test_integration_kit.py -v`
Expected: FAIL — files do not exist.

- [ ] **Step 3: Create the kit files**

```markdown
<!-- owm_server/integration_kit/AGENTS.md -->
# Org World Model (OWM) — grounding rule

Before editing code, changing a config/schema, or modifying infrastructure, call the
OWM MCP tool `owm_ground` (or `owm_blast_radius`) with the change you are about to make
({service | config | path}) and incorporate the predicted blast radius and hidden
couplings into your plan. After a change causes (or avoids) an incident, call
`owm_record_outcome` so the model deepens.
```

```markdown
<!-- owm_server/integration_kit/cursor.owm.mdc -->
---
alwaysApply: true
---
Before any code edit or infra/config change, call the OWM MCP tool `owm_ground`
(or `owm_blast_radius`) for the target {service|config|path} and use the returned
blast radius + couplings before acting.
```

```markdown
<!-- owm_server/integration_kit/cline.owm.md -->
Before editing code or changing config/infra, call `owm_ground` (OWM MCP) for the
target and incorporate the predicted blast radius and hidden couplings. Record what
actually happened afterward with `owm_record_outcome`.
```

```markdown
<!-- owm_server/integration_kit/windsurf.owm.md -->
---
trigger: always_on
---
Before editing code or changing config/infra, call the OWM MCP tool `owm_ground`
for the target {service|config|path} and act on the returned blast radius + couplings.
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/server/test_integration_kit.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `pytest tests/server -v` (Expected: all green) and `pytest -q` (existing `owm/` suite still green — we did not touch `owm/`).

```bash
git add owm_server/integration_kit/ tests/server/test_integration_kit.py
git commit -m "feat(server): portable agent integration kit (AGENTS.md + cursor/cline/windsurf rules)"
```

---

## Self-Review

**1. Spec coverage (MVP cut from §12 + components §3/§9):**
- Persist event-sourced model (③) → Tasks 2,3. ✓
- In-memory projection + inference (②, reuse `propagate`) → Tasks 4,5. ✓
- Grounding pack + Query API (②) → Tasks 5,6. ✓
- MCP server ① + portable rules kit → Tasks 10,11. ✓
- Structure ingestion ④ (code-parse + topology, service-grain, ownership index) → Task 7. ✓
- Manual outcome → surprise-gated deepening ⑥ (reuse `learn.observe`) + ablation → Task 8. ✓
- One process + CLI, one env → Tasks 9 + `env="prod"` default throughout. ✓
- Out of scope held out: ⑤ automated outcome receiver, continuous learning worker + hybrid governance, scale-out, CC hooks, LLM extraction, forgetting, live eval — none built. ✓

**2. Placeholder scan:** every step has runnable code + exact commands + expected output. The three "Note:" blocks point at *existing* `owm/` files to confirm real constant/attribute names (not placeholders — they're reuse-alignment checks for code the engineer can read). No TBD/TODO. ✓

**3. Type consistency:** `Store` method names are used identically across Tasks 4–10 (`upsert_node`, `upsert_edge`, `all_nodes`, `all_edges`, `service_for_path`, `append_incident`, `append_prediction`, `new_version`, `current_version`). `ChangeRef`/`GroundingPack`/`BlastItem` defined in Task 5 and reused unchanged in Tasks 6,8,9,10. `record_outcome` signature defined in Task 8 matches its call sites in Tasks 6,9,10. ✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-23-owm-mvp-walking-skeleton.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
