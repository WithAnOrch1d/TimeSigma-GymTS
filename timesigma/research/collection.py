from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .canonical import ComponentEvidence, ResearchStepEvidence, planning_unit_identity
from .pipeline import NormalizedPipeline


CanonicalEvidence = ComponentEvidence | ResearchStepEvidence


@dataclass(frozen=True)
class EvidenceCollection:
    evidence: tuple[CanonicalEvidence, ...]

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        identities = [item.evidence_id for item in evidence]
        if len(identities) != len(set(identities)):
            raise ValueError("canonical Evidence identities must be unique")
        duplicate_count = _duplicate_canonical_pipeline_count(evidence)
        if duplicate_count:
            raise ValueError("exact task + normalized Pipeline must be unique")
        _validate_research_step_lineage(evidence)
        object.__setattr__(self, "evidence", evidence)

    @classmethod
    def from_directory(cls, root: Path) -> EvidenceCollection:
        directory = root.resolve()
        evidence: list[CanonicalEvidence] = []
        for path in sorted((directory / "evidence" / "components").rglob("evidence.json")):
            evidence.append(ComponentEvidence.from_dict(read_json_mapping(path)))
        for path in sorted(
            (directory / "evidence" / "research-steps").rglob("evidence.json")
        ):
            evidence.append(ResearchStepEvidence.from_dict(read_json_mapping(path)))
        return cls(tuple(evidence))

    def by_id(self, evidence_id: str) -> CanonicalEvidence:
        matches = tuple(item for item in self.evidence if item.evidence_id == evidence_id)
        if len(matches) != 1:
            raise ValueError(f"Evidence was not found uniquely: {evidence_id}")
        return matches[0]

    def find(
        self,
        *,
        kind: str | None = None,
        dataset: str | None = None,
        planning_unit: Mapping[str, Any] | None = None,
        schema: str | None = None,
        normalized_pipeline_id: str | None = None,
    ) -> tuple[CanonicalEvidence, ...]:
        return tuple(
            item
            for item in self.evidence
            if _matches(
                item,
                kind=kind,
                dataset=dataset,
                planning_unit=planning_unit,
                schema=schema,
                normalized_pipeline_id=normalized_pipeline_id,
            )
        )

    def exact_pipeline_match(
        self,
        *,
        task: Mapping[str, Any],
        pipeline: Mapping[str, Any],
    ) -> CanonicalEvidence | None:
        key = canonical_pipeline_key_from_parts(task=task, pipeline=pipeline)
        matches = tuple(item for item in self.evidence if canonical_pipeline_key(item) == key)
        if len(matches) > 1:
            raise ValueError("exact task + normalized Pipeline matched multiple Evidence")
        return matches[0] if matches else None

    def retrieval_candidates(self) -> tuple[CanonicalEvidence, ...]:
        return self.evidence

    def duplicate_canonical_pipeline_count(self) -> int:
        return _duplicate_canonical_pipeline_count(self.evidence)

    def write_to(self, root: Path) -> None:
        directory = root.resolve()
        validate_component_tasks(directory, self.evidence)
        paths: dict[Path, str] = {}
        for item in sorted(self.evidence, key=lambda evidence: evidence.evidence_id):
            path = _evidence_path(directory, item)
            existing = paths.get(path)
            if existing is not None and existing != item.evidence_id:
                raise ValueError(
                    f"canonical Evidence path collision: {existing} and {item.evidence_id}"
                )
            paths[path] = item.evidence_id
            write_json_mapping_if_same(path, item.as_dict())


def validate_component_tasks(root: Path, evidence: tuple[CanonicalEvidence, ...]) -> None:
    """Active additions reuse store tasks; qualification fixtures are not authority."""
    manifest_path = root / "manifest.json"
    components = [item for item in evidence if isinstance(item, ComponentEvidence)]
    if not components or not manifest_path.exists():
        return
    manifest = read_json_mapping(manifest_path)
    if manifest.get("store") != "agent-memory":
        return
    from timesigma.benchmark.zero_shot import load_zero_shot_membership

    membership = {split.dataset: split for split in load_zero_shot_membership(root / manifest["zero_shot_membership"])}
    existing_tasks = {
        json.dumps(read_json_mapping(path)["task"], sort_keys=True)
        for path in (root / "evidence/components").rglob("evidence.json")
    }
    for item in components:
        task = item.as_dict()["task"]
        if task["dataset"] not in manifest["dataset_revisions"] or task["dataset"].casefold() == "aiops":
            raise ValueError("Component dataset is outside the canonical task universe")
        if task["schema"] == "zero_shot":
            split = membership.get(task["dataset"])
            if split is None or task["unit"]["training_curves"] != list(split.training_curves) or task["unit"]["held_out_curves"] != list(split.held_out_curves):
                raise ValueError("Component differs from canonical store-level zero-shot membership")
        elif existing_tasks and json.dumps(task, sort_keys=True) not in existing_tasks:
            raise ValueError("Component task is outside the canonical eligible planning units")
        if task["schema"] == "all_in_one":
            artifact = evidence_directory(root, item) / "artifact.json"
            if artifact.exists():
                metadata = read_json_mapping(artifact)
                if metadata["training_curves"] != task["unit"]["training_curves"] or metadata["test_curves"] != task["unit"]["training_curves"]:
                    raise ValueError("ALL_IN_ONE Component execution order differs; must preserve canonical member order")


def load_evidence_collection(root: Path) -> EvidenceCollection:
    return EvidenceCollection.from_directory(root)


def evidence_directory(root: Path, evidence: CanonicalEvidence) -> Path:
    return _evidence_path(root.resolve(), evidence).parent


def pipeline_identity(evidence: CanonicalEvidence) -> str:
    payload = (
        evidence.pipeline
        if isinstance(evidence, ComponentEvidence)
        else evidence.candidate_pipeline
    )
    return NormalizedPipeline.from_dict(payload).identity_key()


def canonical_pipeline_key(evidence: CanonicalEvidence) -> tuple[str, str, str, str]:
    payload = (
        evidence.pipeline
        if isinstance(evidence, ComponentEvidence)
        else evidence.candidate_pipeline
    )
    return canonical_pipeline_key_from_parts(task=evidence.task, pipeline=payload)


def canonical_pipeline_key_from_parts(
    *,
    task: Mapping[str, Any],
    pipeline: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    return (
        str(task["dataset"]),
        str(task["schema"]),
        json.dumps(
            plain_json(task["unit"]),
            sort_keys=True,
            separators=(",", ":"),
        ),
        NormalizedPipeline.from_dict(pipeline).identity_key(),
    )


def read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Evidence source is missing: {path}") from None
    return as_json_mapping(payload, str(path))


def write_json_mapping_if_same(path: Path, payload: Mapping[str, Any]) -> None:
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"refusing to overwrite different derived output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_json_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_json(item) for item in value]
    if isinstance(value, list):
        return [plain_json(item) for item in value]
    return value


def _duplicate_canonical_pipeline_count(evidence: tuple[CanonicalEvidence, ...]) -> int:
    counts = Counter(canonical_pipeline_key(item) for item in evidence)
    return sum(count - 1 for count in counts.values() if count > 1)


def _validate_research_step_lineage(evidence: tuple[CanonicalEvidence, ...]) -> None:
    by_id = {item.evidence_id: item for item in evidence}
    for item in evidence:
        if not isinstance(item, ResearchStepEvidence):
            continue
        parent = by_id.get(item.baseline_evidence_id)
        if parent is None:
            raise ValueError(
                f"ResearchStepEvidence baseline_evidence_id does not resolve: {item.evidence_id}"
            )
        if not _same_task(parent.task, item.task):
            raise ValueError(
                "ResearchStepEvidence baseline Evidence belongs to another task"
            )
        if pipeline_identity(parent) != NormalizedPipeline.from_dict(
            item.baseline_pipeline
        ).identity_key():
            raise ValueError(
                "ResearchStepEvidence baseline_pipeline does not match baseline Evidence"
            )


def _evidence_path(root: Path, evidence: CanonicalEvidence) -> Path:
    task = evidence.task
    dataset = _safe_path_segment(str(task["dataset"]))
    schema = _schema_directory(str(task["schema"]))
    unit = _unit_segment(str(task["schema"]), task["unit"])
    if isinstance(evidence, ComponentEvidence):
        method = _safe_path_segment(str(evidence.pipeline["method"]))
        return (
            root
            / "evidence"
            / "components"
            / method
            / schema
            / dataset
            / unit
            / "evidence.json"
        )
    experiment_id = evidence.evidence_id.rsplit("/", 1)[-1]
    return (
        root
        / "evidence"
        / "research-steps"
        / dataset
        / schema
        / unit
        / _safe_path_segment(experiment_id)
        / "evidence.json"
    )


def _schema_directory(schema: str) -> str:
    return {
        "naive": "naive",
        "all_in_one": "all-in-one",
        "zero_shot": "zero-shot",
    }[schema]


def _unit_segment(schema: str, unit: Any) -> str:
    if not isinstance(unit, Mapping):
        raise TypeError("planning unit must be a mapping")
    return _safe_path_segment(planning_unit_identity(schema, unit))


def _safe_path_segment(value: str) -> str:
    text = value.strip().replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    if not text:
        raise ValueError("path segment cannot be empty")
    return text


def _matches(
    evidence: CanonicalEvidence,
    *,
    kind: str | None,
    dataset: str | None,
    planning_unit: Mapping[str, Any] | None,
    schema: str | None,
    normalized_pipeline_id: str | None,
) -> bool:
    if kind is not None and evidence.kind != kind:
        return False
    if dataset is not None and evidence.task["dataset"] != dataset:
        return False
    if schema is not None and evidence.task["schema"] != schema:
        return False
    if planning_unit is not None and plain_json(evidence.task["unit"]) != plain_json(planning_unit):
        return False
    if normalized_pipeline_id is not None and pipeline_identity(evidence) != normalized_pipeline_id:
        return False
    return True


def _same_task(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["dataset"] == right["dataset"]
        and left["schema"] == right["schema"]
        and plain_json(left["unit"]) == plain_json(right["unit"])
    )


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


__all__ = [
    "CanonicalEvidence",
    "EvidenceCollection",
    "as_json_mapping",
    "canonical_pipeline_key",
    "canonical_pipeline_key_from_parts",
    "evidence_directory",
    "load_evidence_collection",
    "pipeline_identity",
    "plain_json",
    "read_json_mapping",
    "write_json_mapping_if_same",
]
