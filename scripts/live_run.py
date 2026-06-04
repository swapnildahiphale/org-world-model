"""Throwaway driver for the LIVE cluster run (not part of the tested core).

Given the MEASURED blast set D from injecting EXTRA_LATENCY on productcatalogservice
in the running Online Boutique, it assembles the live ground-truth + incident files
and runs the full predictor matrix (engine vs opus baselines vs BFS) via run_full.

D populates three scored items at once (all the same productcatalog coupling):
  - teach-latency-1   (the incident the engine LEARNS from)   -> incident impacted = D
  - transfer-latency-1 (hidden, scored)                        -> ground truth = D
  - given-pc-handler   (given, scored: a code change to pc)    -> ground truth = D
  - negative-flag-1    (hidden, scored: benign sibling knob)   -> ground truth = []

Usage:
  .venv/bin/python -m scripts.live_run \
     --degraded frontend,recommendationservice,checkoutservice \
     --repo-manifest /path/to/deployed-or-release/kubernetes-manifests.yaml \
     --provider claude-code --out results_live
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from scripts.run import run_full
from owm.baselines import make_live_client

ROOT = Path(__file__).resolve().parents[1]


def _stage_repo(manifest_path: Path) -> Path:
    """parse_repo wants <root>/kubernetes-manifests.yaml. Stage a dir pointing at
    the real OB manifest so the static parse reflects the actually-running topology.

    The real release manifest declares DISABLE_PROFILER (-> the BENIGN_KNOB config
    node parses for free) but NOT EXTRA_LATENCY. EXTRA_LATENCY is a genuine, runtime
    env knob of productcatalogservice (the one we inject during the fault). We declare
    it here in its OFF state (value "0s"), exactly as DISABLE_PROFILER is declared, so
    the static parser emits `productcatalogservice::EXTRA_LATENCY` as a real, inert,
    unconnected config node. The parser still cannot know it couples to anything --
    that downstream blast is the hidden edge the engine must LEARN. This keeps Rule 2
    (taught node = genuinely-parsed node) honest for the live topology.
    """
    import yaml

    docs = list(yaml.safe_load_all(manifest_path.read_text()))
    for doc in docs:
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        if doc.get("metadata", {}).get("name") != "productcatalogservice":
            continue
        containers = doc["spec"]["template"]["spec"]["containers"]
        env = containers[0].setdefault("env", [])
        names = {e.get("name") for e in env}
        if "EXTRA_LATENCY" not in names:
            env.append({"name": "EXTRA_LATENCY", "value": "0s"})

    d = Path(tempfile.mkdtemp(prefix="owm_live_repo_"))
    (d / "kubernetes-manifests.yaml").write_text(
        "\n---\n".join(yaml.safe_dump(doc, sort_keys=False) for doc in docs if doc)
    )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded", required=True,
                    help="comma-separated services measured to degrade under EXTRA_LATENCY (set D)")
    ap.add_argument("--repo-manifest", required=True,
                    help="path to the OB kubernetes manifest to statically parse")
    ap.add_argument("--provider", default="claude-code", choices=["anthropic", "claude-code"])
    ap.add_argument("--self-consistency-n", type=int, dest="n", default=5)
    ap.add_argument("--out", default="results_live")
    args = ap.parse_args()

    D = sorted({s.strip() for s in args.degraded.split(",") if s.strip()})
    print(f">>> measured blast set D = {D}")

    data = ROOT / "data" / "groundtruth"
    data.mkdir(parents=True, exist_ok=True)

    gt = [
        {"scenario_id": "given-pc-handler", "stratum": "given", "impacted": D},
        {"scenario_id": "transfer-latency-1", "stratum": "hidden", "impacted": D},
        {"scenario_id": "negative-flag-1", "stratum": "hidden", "impacted": []},
    ]
    gt_path = data / "ground_truth_live.json"
    gt_path.write_text(json.dumps(gt, indent=2))

    incidents = [{"origin": "productcatalogservice::EXTRA_LATENCY",
                  "impacted": D, "scenario_id": "teach-latency-1"}]
    inc_path = data / "incidents_live.json"
    inc_path.write_text(json.dumps(incidents, indent=2))

    repo = _stage_repo(Path(args.repo_manifest))
    print(f">>> staged repo for parse_repo at {repo}")

    client = make_live_client(provider=args.provider)
    res = run_full(offline=False, repo_path=repo,
                   ground_truth_path=gt_path, incidents_path=inc_path,
                   out_dir=ROOT / args.out, llm_client=client,
                   self_consistency_n=args.n)
    print(json.dumps(res, indent=2))
    print("\n>>> HIDDEN-stratum Brier by predictor:", res["hidden_brier"])


if __name__ == "__main__":
    main()
