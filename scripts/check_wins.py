"""Judge the pre-registered Week-1 win conditions from a results dict. Honest:
returns a bool per condition; prints PASS/FAIL. A FALSE is a finding to report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EPS = 1e-9


def evaluate_wins(results: dict) -> dict[str, bool]:
    hb = results["hidden_brier"]
    deep = results["deepening"]
    contrast = results.get("state_contrast", {})
    return {
        "engine_beats_baselines": all(
            hb["engine"] < hb[b] for b in ("stateless_llm", "rag_llm", "bfs")
        ),
        "engine_deepens": deep["engine"]["brier_post"] < deep["engine"]["brier_pre"],
        "ablation_flat":
            deep["engine_no_learning"]["brier_post"]
            >= deep["engine_no_learning"]["brier_pre"] - EPS,
        "state_amplifies": any(v > 0 for v in contrast.values()) if contrast else False,
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/results.json")
    data = json.loads(path.read_text())
    if data.get("mode") == "offline-stub":
        print("WARNING: offline-stub run — the baseline comparison is NOT substantive.")
        print("         Offline, trust only the deepening + ablation signals.\n")
    wins = evaluate_wins(data)
    for k, v in wins.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    sys.exit(0 if all(wins.values()) else 1)


if __name__ == "__main__":
    main()
