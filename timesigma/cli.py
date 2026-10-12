from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .algorithms import EASYTSAD_SCHEMAS, METHOD_CATALOG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timesigma",
        description="Run audited EasyTSAD methods through the native experiment lifecycle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "catalog",
        help="Show native Schema support for the audited EasyTSAD methods.",
    )

    run = commands.add_parser(
        "run",
        help="Run one audited method and produce native EasyTSAD results.",
    )
    run.add_argument("--method", required=True, choices=tuple(METHOD_CATALOG))
    run.add_argument("--schema", required=True, choices=EASYTSAD_SCHEMAS)
    run.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="EasyTSAD dataset family; repeat to run more than one.",
    )
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--seed", type=int, default=1)

    run.add_argument(
        "--zero-shot-membership",
        type=Path,
        help="Reuse a frozen zero-shot curve split from an earlier run.",
    )

    capabilities = commands.add_parser(
        "capabilities",
        help="Run the planned non-AIOPS EasyTSAD capability experiments.",
    )
    capabilities.add_argument("--plan", type=Path, required=True)
    capabilities.add_argument("--dataset-root", type=Path, required=True)
    capabilities.add_argument("--source-root", type=Path, action="append")
    capabilities.add_argument("--output-dir", type=Path, required=True)
    capabilities.add_argument(
        "--method", dest="methods", action="append", choices=tuple(METHOD_CATALOG)
    )
    capabilities.add_argument(
        "--schema", dest="schemas", action="append", choices=EASYTSAD_SCHEMAS
    )
    capabilities.add_argument("--dataset", dest="datasets", action="append")

    leaderboard = commands.add_parser(
        "leaderboard",
        help="Read an official EasyTSAD Summary as a local metric table.",
    )
    leaderboard.add_argument(
        "--results-dir", type=Path, action="append", required=True
    )
    leaderboard.add_argument("--schema", required=True, choices=EASYTSAD_SCHEMAS)
    leaderboard.add_argument("--dataset", required=True)
    leaderboard.add_argument("--metric", required=True)

    agent_memory = commands.add_parser(
        "agent-memory-bootstrap",
        help="Run official/default Components into a local bootstrap root and materialize Agent Memory.",
    )
    agent_memory.add_argument("--output-dir", type=Path, required=True)
    agent_memory.add_argument(
        "--bootstrap-root",
        type=Path,
        help="Run-local EasyTSAD execution root; defaults beside output-dir.",
    )
    agent_memory.add_argument(
        "--dataset-root",
        dest="dataset_roots",
        type=Path,
        action="append",
        required=True,
        help="Clean dataset source root; repeat for NEK or other override roots.",
    )
    agent_memory.add_argument("--profile-root", type=Path, default=Path("runs/profiles"))
    agent_memory.add_argument(
        "--method", dest="methods", action="append", choices=tuple(METHOD_CATALOG)
    )
    agent_memory.add_argument(
        "--schema", dest="schemas", action="append", choices=EASYTSAD_SCHEMAS
    )
    agent_memory.add_argument("--dataset", dest="datasets", action="append")
    agent_memory.add_argument("--seed", type=int, default=1)

    validate_memory = commands.add_parser(
        "agent-memory-validate",
        help="Validate active Agent Memory store integrity and artifact reload.",
    )
    validate_memory.add_argument("--root", type=Path, default=Path("runs/agent-memory"))
    validate_memory.add_argument(
        "--skip-artifact-load",
        action="store_true",
        help="Check paths and metadata without loading checkpoint/estimator bytes.",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)

    if args.command == "catalog":
        return {
            "command": "catalog",
            "methods": {
                name: {
                    schema: (
                        "supported"
                        if spec.capability(schema).supported
                        else "N/A"
                    )
                    for schema in EASYTSAD_SCHEMAS
                }
                for name, spec in METHOD_CATALOG.items()
            },
        }

    if args.command == "leaderboard":
        from .benchmark.leaderboard import build_leaderboard

        return {
            "command": "leaderboard",
            **build_leaderboard(
                results_dirs=tuple(args.results_dir),
                schema=args.schema,
                dataset=args.dataset,
                metric_id=args.metric,
            ),
        }

    if args.command == "capabilities":
        from .benchmark.capabilities import run_capabilities

        return {
            "command": "capabilities",
            **run_capabilities(
                plan_path=args.plan,
                dataset_root=args.dataset_root,
                source_roots=tuple(args.source_root or ()),
                output_dir=args.output_dir,
                methods=tuple(args.methods or ()),
                schemas=tuple(args.schemas or ()),
                datasets=tuple(args.datasets or ()),
            ),
        }

    if args.command == "agent-memory-bootstrap":
        from .research import materialize_agent_memory

        return {
            "command": "agent-memory-bootstrap",
            **materialize_agent_memory(
                output_dir=args.output_dir,
                dataset_roots=tuple(args.dataset_roots),
                profile_root=args.profile_root,
                bootstrap_root=args.bootstrap_root,
                methods=tuple(args.methods or ()),
                schemas=tuple(args.schemas or ()),
                datasets=tuple(args.datasets or ()),
                seed=args.seed,
            ).as_dict(),
        }

    if args.command == "agent-memory-validate":
        from .research import validate_agent_memory_store

        return {
            "command": "agent-memory-validate",
            **validate_agent_memory_store(
                args.root,
                load_artifacts=not args.skip_artifact_load,
            ).as_dict(),
        }

    from .benchmark.runtime import run_method

    results_dir, runtime_python = run_method(
        method_name=args.method,
        schema=args.schema,
        datasets=tuple(args.datasets),
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        zero_shot_membership=args.zero_shot_membership,
        seed=args.seed,
    )
    outcome = {
        "command": "run",
        "method": args.method,
        "schema": args.schema,
        "datasets": args.datasets,
        "results_dir": str(results_dir),
        "runtime_python": str(runtime_python),
        "run_manifest_path": str(args.output_dir / "run.json"),
    }
    if args.schema == "zero_shot":
        outcome["zero_shot_membership_path"] = str(
            results_dir / "zero_shot_membership.json"
        )
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        build_parser().print_help()
        return 0

    try:
        outcome = run_cli(arguments)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {"passed": True, **outcome},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_cli"]
