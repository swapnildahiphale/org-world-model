"""Baseline predictors, each neutralizing a specific rebuttal (spec §7):
  StatelessLLM  - "did you even give the LLM the structure?"  (the floor)
  RAGLLM        - "it's just retrieval"   (the on-thesis opponent)
  BFSBaseline   - "a dumb graph walk does this"  (isolates calibration value)

LLM confidence = self-consistency frequency over n samples (NOT a verbalized
'I'm 80% sure', which would be a strawman). The client is injected so the whole
matrix runs offline with FakeClient; live runs pin MODEL_ID at temp>0.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

from owm.groundtruth import Prediction

# Pinned for the record (PREREGISTRATION.md), overridable via env for cheap test
# runs (e.g. OWM_LLM_MODEL=claude-haiku-4-5-20251001).
MODEL_ID = os.environ.get("OWM_LLM_MODEL", "claude-opus-4-8")
SELF_CONSISTENCY_N = 5


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class FakeClient:
    """Deterministic offline client. Cycles through canned responses so a test
    can simulate self-consistency variance."""

    def __init__(self, responses):
        self._responses = responses if isinstance(responses, list) else [responses]
        self._i = 0

    def complete(self, prompt: str) -> str:
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r


_JSON_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)
_FORMAT = ('\n\nRespond with ONLY a JSON array of the service names you predict '
           'will be in the blast radius, e.g. ["frontend","cartservice"].')


def parse_service_set(text: str, services) -> set[str]:
    m = _JSON_ARRAY.search(text)
    if not m:
        return set()
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return set()
    known = set(services)
    return {s for s in arr if s in known}


class _LLMBaseline:
    include_incidents = False

    def __init__(self, client: LLMClient, n: int = SELF_CONSISTENCY_N, model_id: str = MODEL_ID):
        self.client, self.n, self.model_id = client, n, model_id

    def predict(self, scenario, pack) -> Prediction:
        from owm.context import as_llm_text  # local import avoids a cycle
        prompt = as_llm_text(pack, include_incidents=self.include_incidents) + _FORMAT
        services = pack["services"]
        counts = {s: 0 for s in services}
        for _ in range(self.n):
            for s in parse_service_set(self.client.complete(prompt), services):
                counts[s] += 1
        probs = {s: counts[s] / self.n for s in services
                 if s not in scenario.touched_services}
        return Prediction(scenario.id, probs)


class StatelessLLM(_LLMBaseline):
    include_incidents = False


class RAGLLM(_LLMBaseline):
    include_incidents = True


class BFSBaseline:
    """Deterministic reachability over the GIVEN call graph. Every transitive
    caller of a touched service is in-blast with probability 1.0. No notion of
    config couplings => predicts 0 for config-class changes."""

    def __init__(self, cwm):
        self.cwm = cwm

    def predict(self, scenario, pack=None) -> Prediction:
        probs = {s: 0.0 for s in self.cwm.services}
        frontier = list(scenario.touched_services)
        seen = set(frontier)
        while frontier:
            node = frontier.pop()
            for caller, _w in self.cwm.callers_of(node):
                if caller not in seen:
                    seen.add(caller)
                    frontier.append(caller)
                    probs[caller] = 1.0
        for s in scenario.touched_services:
            probs.pop(s, None)
        return Prediction(scenario.id, probs)


class ClaudeCodeClient:
    """Routes baseline LLM calls through the Claude Code CLI (`claude -p`) instead
    of the Anthropic API key — it authenticates via the user's Claude Code
    subscription, so these calls do NOT consume API-key quota (load-balancing).

    Trade-off vs the API client: no temperature/seed control and a heavier
    per-call cost (each call spawns a headless agent), so it is great for cheap
    test runs but less reproducible than a pinned API model for a final result.
    """

    def __init__(self, model: str | None = None, timeout: int = 300):
        self.model = model or MODEL_ID
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        import subprocess

        proc = subprocess.run(
            ["claude", "-p", prompt, "--model", self.model, "--output-format", "text"],
            capture_output=True, text=True, timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({proc.returncode}): {proc.stderr[:300]}")
        return proc.stdout


def make_live_client(provider: str = "anthropic", temperature: float = 0.7) -> LLMClient:
    """Build a real client for live runs (T15). Imported lazily so offline runs
    never need the SDK or an API key.

    provider="anthropic"   -> Anthropic API (needs ANTHROPIC_API_KEY).
    provider="claude-code" -> the `claude` CLI (uses Claude Code auth, no API key).
    Both honour OWM_LLM_MODEL for the model id.
    """
    if provider == "anthropic":
        import anthropic

        class _Anthropic:
            def __init__(self):
                self._c = anthropic.Anthropic()

            def complete(self, prompt: str) -> str:
                msg = self._c.messages.create(
                    model=MODEL_ID, max_tokens=512, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

        return _Anthropic()
    if provider == "claude-code":
        return ClaudeCodeClient()
    raise ValueError(f"unknown provider {provider!r}")
