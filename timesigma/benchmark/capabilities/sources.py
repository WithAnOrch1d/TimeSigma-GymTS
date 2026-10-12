from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterator, Mapping, Sequence

from ..manifest import dataset_provenance


@dataclass(frozen=True)
class DatasetSource:
    root: Path
    provenance: Mapping[str, object]
    relative_path: str


def resolve_dataset_sources(
    plan: Mapping[str, object],
    dataset_root: Path,
    source_roots: Sequence[Path],
    datasets: Sequence[str],
) -> dict[str, DatasetSource]:
    default = _provenance(dataset_root)
    if not _matches(default, plan["dataset_source"], plan["dataset_revision"]):
        raise ValueError("capability dataset source must match the clean frozen revision")

    overrides = plan["dataset_sources"]
    if not isinstance(overrides, Mapping):
        raise TypeError("capability dataset_sources must be a mapping")
    additional = [(root.resolve(), _provenance(root)) for root in source_roots]
    declared = {
        (source["source"], source["revision"])
        for source in overrides.values()
        if isinstance(source, Mapping)
    }
    if any(
        provenance.get("clean") is not True
        or (provenance.get("source"), provenance.get("revision")) not in declared
        for _, provenance in additional
    ):
        raise ValueError("capability source root must match a clean frozen revision")

    resolved: dict[str, DatasetSource] = {}
    for dataset in datasets:
        source = overrides.get(dataset)
        if source is None:
            resolved[dataset] = DatasetSource(
                root=dataset_root,
                provenance=default,
                relative_path=f"UTS/{dataset}",
            )
            continue
        if not isinstance(source, Mapping):
            raise TypeError(f"capability dataset source must be a mapping: {dataset}")
        matches = [
            (root, provenance)
            for root, provenance in additional
            if _matches(provenance, source["source"], source["revision"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"capability source root is missing or ambiguous for dataset {dataset}"
            )
        root, provenance = matches[0]
        resolved[dataset] = DatasetSource(
            root=root,
            provenance=provenance,
            relative_path=str(source["relative_path"]),
        )
    return resolved


@contextmanager
def dataset_view(sources: Mapping[str, DatasetSource]) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="timesigma-capability-data-") as directory:
        root = Path(directory)
        uts = root / "UTS"
        uts.mkdir()
        for dataset, source in sources.items():
            source_root = source.root.resolve()
            target = (source_root / source.relative_path).resolve()
            if not target.is_relative_to(source_root):
                raise ValueError(f"capability dataset path escapes its source: {dataset}")
            if not target.is_dir():
                raise FileNotFoundError(f"capability dataset is missing: {target}")
            (uts / dataset).symlink_to(target, target_is_directory=True)
        yield root


def _provenance(root: Path) -> Mapping[str, object]:
    return dataset_provenance(root.resolve())


def _matches(provenance: Mapping[str, object], source: object, revision: object) -> bool:
    return (
        provenance.get("source") == source
        and provenance.get("revision") == revision
        and provenance.get("clean") is True
    )


__all__ = ["DatasetSource", "dataset_view", "resolve_dataset_sources"]
