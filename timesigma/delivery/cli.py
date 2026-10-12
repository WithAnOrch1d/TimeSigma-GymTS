from __future__ import annotations

import argparse
import json
from pathlib import Path

from timesigma.research.collection import write_json_mapping_if_same

from .execution import bounded_process, evaluate_all, execute_all, execute_worker, leaderboard, seal_scores
from .planning import close_planning
from .protocol import REPOSITORY, load_protocol, prepare_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen plan-all / score-all / evaluate delivery; stages never auto-advance")
    parser.add_argument("stage", choices=("prepare", "close-planning", "execute", "freeze-scores", "evaluate", "leaderboard", "worker", "evaluate-worker"))
    parser.add_argument("root", type=Path)
    parser.add_argument("extra", nargs="*")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        if args.config is None:
            parser.error("prepare requires --config")
        result = prepare_protocol(args.config, root)
        print(json.dumps({"status": result["status"], "targets": len(result["curves"]), "root": str(root)}))
        return
    if args.stage == "worker":
        execute_worker(root, args.extra[0], args.extra[1], int(args.extra[2]))
        return
    if args.stage == "evaluate-worker":
        evaluate_all(root)
        return
    if args.stage == "evaluate":
        protocol = load_protocol(root)
        if (root / "evaluation-complete.json").exists():
            raise ValueError("evaluation is already complete")
        attempts = sorted(root.glob("evaluation-attempt-*.json"))
        if attempts and not args.retry_failed:
            raise ValueError("evaluation previously failed; an explicit same-score engineering retry is required")
        attempt = len(attempts) + 1
        if attempt > protocol["evaluation"]["max_attempts"]:
            raise ValueError("evaluation retry budget exhausted")
        result = bounded_process(
            [str(REPOSITORY / ".venv/bin/python"), "-m", "timesigma.delivery.cli", "evaluate-worker", str(root)],
            root / f"evaluation-{attempt}.log", seconds=protocol["evaluation"]["max_wall_seconds"],
            memory_mb=protocol["evaluation"]["memory_mb"], cpu_percent=protocol["execution"]["cpu_percent"],
            backend=protocol.get("resource_backend", "systemd"),
        )
        write_json_mapping_if_same(root / f"evaluation-attempt-{attempt}.json", result)
        if result["returncode"] != 0:
            raise RuntimeError("official evaluation failed; preserve scores and inspect the evaluation log")
        write_json_mapping_if_same(root / "evaluation-complete.json", result)
    else:
        functions = {"close-planning": close_planning, "freeze-scores": seal_scores, "leaderboard": leaderboard}
        result = execute_all(root, retry_failed=args.retry_failed) if args.stage == "execute" else functions[args.stage](root)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
