from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_current_modules_import_without_deleted_compat_modules() -> None:
    code = """
import importlib
import json
import sys

modules = (
    "timesigma.research.result_parser",
    "timesigma.research.decision",
    "timesigma.research.canonical",
    "timesigma.research.agent_memory",
    "timesigma.research.landscape",
    "timesigma.research.state",
    "timesigma.research.provider_context",
    "timesigma.research.tools",
)
for module in modules:
    importlib.import_module(module)

forbidden = (
    "timesigma.research.cases",
    "timesigma.research.conversion",
    "timesigma.research.evidence",
    "timesigma.research.legacy_sources",
    "timesigma.research.memory",
    "timesigma.research.legacy",
    "timesigma.research.legacy.canonical",
    "timesigma.research.legacy.cases",
    "timesigma.research.legacy.conversion",
    "timesigma.research.legacy.evidence",
    "timesigma.research.legacy.history",
    "timesigma.research.legacy.memory",
    "timesigma.research.legacy.sources",
)
print(json.dumps({
    "forbidden_loaded": [name for name in forbidden if name in sys.modules],
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"forbidden_loaded": []}
    assert completed.stderr == ""


def test_current_research_public_exports_do_not_expose_legacy_case_memory() -> None:
    import timesigma.research as research

    forbidden = {
        "ComponentCase",
        "PipelineCase",
        "ExperimentMemory",
        "build_component_cases",
        "rebuild_experiment_memory",
        "convert_legacy_evidence",
        "build_component_evidence",
        "build_research_step_evidence",
        "read_research_decision",
        "write_research_decision",
    }
    assert forbidden.isdisjoint(research.__all__)
    assert all(not hasattr(research, name) for name in forbidden)


def test_current_tools_do_not_expose_legacy_case_memory_apis() -> None:
    code = """
import json

import timesigma.research.tools as tools

legacy_names = (
    "ComponentCase",
    "PipelineCase",
    "ExperimentMemory",
    "build_component_cases",
    "rebuild_experiment_memory",
    "convert_legacy_evidence",
)
print(json.dumps({"exposed": [name for name in legacy_names if hasattr(tools, name)]}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"exposed": []}
    assert completed.stderr == ""
