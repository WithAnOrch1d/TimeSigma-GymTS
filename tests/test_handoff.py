"""The handoff must work without the original checkout or scientific store."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import timesigma


ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_example_uses_only_shipped_code_and_assets(capsys, monkeypatch):
    from timesigma.benchmark import runtime

    def forbidden_training(*args, **kwargs):
        raise AssertionError("read-only handoff example must not run a detector")

    monkeypatch.setattr(runtime, "run_method", forbidden_training)
    example = load_example("handoff_inspect", "examples/handoff/inspect_memory.py")
    example.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS"
    assert result["validation"]["component_evidence"] == 4
    assert result["validation"]["research_step_evidence"] == 2
    assert result["comparison"]["coverage"]["missing"] == 1
    for name, module in tuple(sys.modules.items()):
        if name == "timesigma" or name.startswith("timesigma."):
            assert Path(module.__file__).resolve().is_relative_to(ROOT / "timesigma")
    assert not (ROOT / "timesigma/research/component_queue.py").exists()


def test_prepare_rejects_unbound_dataset_before_copying_memory(tmp_path):
    example = load_example("handoff_prepare", "examples/prepare_research.py")
    dataset = tmp_path / "unbound-data"
    dataset.mkdir()
    output = tmp_path / "demo"
    with pytest.raises(ValueError, match="clean pinned dataset"):
        example.prepare(dataset, output)
    assert not output.exists()
