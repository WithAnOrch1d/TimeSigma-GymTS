from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import timesigma
from EasyTSAD.Controller import TSADController as _TSADController  # noqa: F401
from EasyTSAD.Methods import BaseMethod


def test_public_api_exposes_only_the_native_easytsad_surface() -> None:
    assert issubclass(timesigma.TimeSigma, BaseMethod)
    assert timesigma.get_method("LOF").method_name == "LOF"
    assert timesigma.EASYTSAD_SCHEMAS == ("naive", "all_in_one", "zero_shot")

    removed_names = {
        "ALGORITHMS",
        "SKILLS",
        "T0Executor",
        "T1Executor",
        "ExperimentRunner",
        "IsolatedEvaluator",
        "load_active_registry",
    }
    assert removed_names.isdisjoint(timesigma.__all__)
    assert all(not hasattr(timesigma, name) for name in removed_names)


def test_importing_cli_does_not_load_the_retired_execution_stack() -> None:
    root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import timesigma.cli

retired = (
    'timesigma.active_registry',
    'timesigma.agent.orchestrator',
    'timesigma.agent.router',
    'timesigma.benchmark.evaluator',
    'timesigma.benchmark.g5b',
    'timesigma.benchmark.g5b_pi',
    'timesigma.benchmark.g5b_runner',
    'timesigma.protocol_candidates',
    'timesigma.qualification',
    'timesigma.skills',
)
print(json.dumps({
    'retired': [name for name in retired if name in sys.modules],
    'easytsad': [name for name in sys.modules if name == 'EasyTSAD' or name.startswith('EasyTSAD.')],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"retired": [], "easytsad": []}
    assert completed.stderr == ""
