"""Prepare and validate one NAIVE Research scope, without training or an LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys
import subprocess

from timesigma.benchmark.manifest import dataset_provenance
from timesigma.research.tools import dispatch_research_tool


ROOT = Path(__file__).resolve().parents[1]
REVISION = "0c6dfa195a5af8ef75938bc1ca6b870e49c7b14a"
SOURCE = "https://github.com/CSTCloudOps/datasets.git"


def prepare(dataset_source: Path, output: Path) -> dict:
    dataset_source, output = dataset_source.resolve(), output.resolve()
    provenance = dataset_provenance(dataset_source)
    if (provenance["source"] != SOURCE or provenance["revision"] != REVISION
            or provenance["clean"] is not True):
        raise ValueError("first obtain the clean pinned dataset with scripts/fetch_example_data.py")
    dataset_dir = dataset_source / "UTS/TODS"
    if not (dataset_dir / "234/train.npy").is_file():
        raise FileNotFoundError("TODS/234 is missing")
    # A new local working store is needed because Research commits new steps.
    # The shipped example and any canonical store must remain unchanged.
    if output.exists():
        raise FileExistsError("use a new demo output directory; previous attempts are preserved")
    output.mkdir(parents=True)
    shutil.copytree(ROOT / "examples/handoff/sample", output / "memory")
    scope = {
        "dataset": "TODS", "schema": "naive", "target_metric": "point-wise-f1",
        "planning_unit": {"curve": "234", "training_curves": ["234"], "held_out_curves": []},
        "dataset_root": str(dataset_source), "dataset_source_root": str(dataset_source),
        "dataset_dir": str(dataset_dir), "dataset_source": SOURCE, "dataset_revision": REVISION,
        "profile_dir": str(output / "memory/profiles/TODS" / REVISION),
        "zero_shot_membership": None,
        "agent_memory_root": str(output / "memory/agent-memory"), "memory_mode": "on",
    }
    scope_path = output / "scope.json"
    scope_path.write_text(json.dumps(scope, indent=2) + "\n")
    # Exercise the real context/data/Memory boundary. No experiment tool is called.
    context = dispatch_research_tool("research.context", {
        **scope, "experiment_root": str(output / "run/experiments"),
        "experiment_index": 1, "max_experiments": 1, "max_wall_seconds": 1200,
    })
    (output / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    command = [
        "node", f"--env-file-if-exists={ROOT / 'pi-host/.env'}",
        str(ROOT / "pi-host/dist/research-cli.js"),
        "--scope", str(scope_path), "--target-metric", "point-wise-f1",
        "--max-experiments", "1", "--max-wall-seconds", "1200",
        "--max-provider-calls", "24", "--max-provider-tokens", "200000",
        "--provider-log", str(output / "run/provider.jsonl"),
        "--session-dir", str(output / "run/sessions"),
        "--repository-root", str(ROOT), "--python", sys.executable,
    ]
    launch = {"cwd": str(ROOT), "argv": command,
              "note": "Run explicitly to call the configured Provider and allow at most one experiment."}
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    return {"status": "PREPARED_NOT_STARTED", "scope": str(scope_path),
            "command": shlex.join(command),
            "note": "Agent may finish without a new experiment. Core12 and its runtime requirements remain unchanged."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-source", type=Path, default=ROOT / "datasets/easytsad")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/demo-001")
    parser.add_argument('--execute', action='store_true', help='After preparation, run the real paid Agent session')
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_source, args.output), ensure_ascii=False, indent=2))
    if args.execute:
        launch = json.loads((args.output.resolve() / 'launch.json').read_text())
        subprocess.run(launch['argv'], cwd=launch['cwd'], check=True)


if __name__ == "__main__":
    main()
