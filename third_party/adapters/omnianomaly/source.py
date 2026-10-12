"""Exact, unmodified author sources and the isolated numeric worker runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[2]
UPSTREAMS = REPO / "third_party/upstreams/omnianomaly"
RUNTIME_DIRECTORY = REPO / "third_party/runtimes/omnianomaly"
PYTHON = REPO / ".runtimes/omnianomaly/bin/python"


def source_binding():
    sources = json.loads((DIRECTORY / "sources.json").read_text())
    for source in sources:
        root = UPSTREAMS / source["name"]
        for name, expected in source["files"].items():
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("OmniAnomaly pinned source missing or changed: " + str(path))
        # No undeclared Python module may shadow an audited dependency.
        actual = {str(p.relative_to(root)) for p in root.rglob("*.py")}
        if actual != {name for name in source["files"] if name.endswith(".py")}:
            raise ValueError("OmniAnomaly source contains undeclared Python files")
    return sources


def adapter_binding():
    # Bind executable adapter/runtime bytes at the Artifact reload boundary.
    # The compatibility manifest cannot include its own digest recursively.
    return {prefix + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for directory, prefix in ((DIRECTORY, ""), (RUNTIME_DIRECTORY, "runtime/"))
            for p in sorted(directory.iterdir())
            if p.suffix in (".py", ".toml", ".json", ".txt") and p.name != "replay-compatibility.json"}


def validate_adapter_binding(recorded):
    current = adapter_binding()
    if recorded == current:
        return
    # Accept only the qualified pre-move identity paired with these exact
    # relocated bytes. Future edits cannot inherit this historical allowance.
    migration = json.loads((DIRECTORY / "replay-compatibility.json").read_text())
    if recorded != migration["qualified_binding"] or current != migration["relocated_binding"]:
        raise ValueError("OmniAnomaly Artifact source/adapter differs")
