from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

from timesigma.benchmark.leaderboard import OFFICIAL_METRIC_IDS

from .canonical import ComponentEvidence
from .collection import EvidenceCollection, plain_json
from .pipeline import TaskContext


@dataclass(frozen=True)
class ComponentLandscape:
    task: TaskContext
    components: tuple[ComponentEvidence, ...]
    best_value: float | None
    best_evidence_ids: tuple[str, ...]
    default_baseline_evidence_id: str | None

    @classmethod
    def from_collection(
        cls,
        collection: EvidenceCollection,
        task: TaskContext,
    ) -> ComponentLandscape:
        components = tuple(
            evidence
            for evidence in collection.evidence
            if isinstance(evidence, ComponentEvidence)
            and _matches_task(evidence, task)
        )
        _require_unique_method_per_task(components)
        ordered = tuple(
            sorted(
                components,
                key=lambda evidence: (
                    -_target_value(evidence, task),
                    evidence.evidence_id,
                ),
            )
        )
        if not ordered:
            return cls(
                task=task,
                components=(),
                best_value=None,
                best_evidence_ids=(),
                default_baseline_evidence_id=None,
            )

        best_value = _target_value(ordered[0], task)
        best_ids = tuple(
            evidence.evidence_id
            for evidence in ordered
            if _target_value(evidence, task) == best_value
        )
        return cls(
            task=task,
            components=ordered,
            best_value=best_value,
            best_evidence_ids=best_ids,
            default_baseline_evidence_id=min(best_ids),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.as_dict(),
            "components": [component.as_dict() for component in self.components],
            "best_value": self.best_value,
            "best_evidence_ids": list(self.best_evidence_ids),
            "default_baseline_evidence_id": self.default_baseline_evidence_id,
        }


def build_component_landscape(
    collection: EvidenceCollection,
    task: TaskContext,
) -> ComponentLandscape:
    return ComponentLandscape.from_collection(collection, task)


def _matches_task(evidence: ComponentEvidence, task: TaskContext) -> bool:
    return (
        evidence.task["dataset"] == task.dataset
        and evidence.task["schema"] == task.schema
        and plain_json(evidence.task["unit"]) == plain_json(task.unit.as_dict())
    )


def _target_value(evidence: ComponentEvidence, task: TaskContext) -> float:
    if task.target_metric not in OFFICIAL_METRIC_IDS:
        raise ValueError(f"unsupported EasyTSAD leaderboard metric: {task.target_metric}")
    value = evidence.metrics.get(task.target_metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ComponentEvidence is missing target metric: {evidence.evidence_id}")
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"ComponentEvidence target metric is not finite: {evidence.evidence_id}")
    return metric


def _require_unique_method_per_task(
    components: tuple[ComponentEvidence, ...],
) -> None:
    counts = Counter(str(component.pipeline["method"]) for component in components)
    duplicates = sorted(method for method, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "duplicate ComponentEvidence method for exact task: "
            + ", ".join(duplicates)
        )


__all__ = ["ComponentLandscape", "build_component_landscape"]
