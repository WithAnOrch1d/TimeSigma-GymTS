from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

from .canonical import ComponentEvidence, ResearchStepEvidence
from .collection import EvidenceCollection, pipeline_identity, plain_json
from .landscape import ComponentLandscape
from .pipeline import TaskContext


@dataclass(frozen=True)
class ResearchRuntimeSnapshot:
    experiments_used: int
    max_experiments: int
    max_wall_seconds: int | None
    wall_seconds_used: float | None = None
    model_finish_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.experiments_used, bool) or self.experiments_used < 0:
            raise ValueError("experiments_used must be a non-negative integer")
        if isinstance(self.max_experiments, bool) or self.max_experiments <= 0:
            raise ValueError("max_experiments must be a positive integer")
        if self.experiments_used > self.max_experiments:
            raise ValueError("experiments_used cannot exceed max_experiments")
        if self.max_wall_seconds is not None and (
            isinstance(self.max_wall_seconds, bool) or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be positive when provided")
        if self.wall_seconds_used is not None:
            wall = float(self.wall_seconds_used)
            if not math.isfinite(wall) or wall < 0:
                raise ValueError("wall_seconds_used must be finite and non-negative")
            object.__setattr__(self, "wall_seconds_used", wall)
        if self.model_finish_reason is not None and not self.model_finish_reason.strip():
            raise ValueError("model_finish_reason cannot be empty")

    @property
    def experiments_remaining(self) -> int:
        return self.max_experiments - self.experiments_used

    @property
    def experiment_index(self) -> int:
        return self.experiments_used + 1

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "experiment_index": self.experiment_index,
            "experiments_used": self.experiments_used,
            "max_experiments": self.max_experiments,
            "experiments_remaining": self.experiments_remaining,
            "max_wall_seconds": self.max_wall_seconds,
        }
        if self.wall_seconds_used is not None:
            payload["wall_seconds_used"] = self.wall_seconds_used
        return payload


@dataclass(frozen=True)
class ResearchState:
    task: TaskContext
    initial_baseline: Mapping[str, Any] | None
    global_best_value: float | None
    global_best_evidence_ids: tuple[str, ...]
    global_incumbent_evidence_id: str | None
    executed_pipelines: Mapping[str, str]
    research_step_count: int
    research_step_evidence_ids: tuple[str, ...]
    lineage_edges: tuple[Mapping[str, str], ...]
    runtime_budget: Mapping[str, Any]
    stop_status: Mapping[str, Any]

    @classmethod
    def from_evidence(
        cls,
        collection: EvidenceCollection,
        landscape: ComponentLandscape,
        runtime: ResearchRuntimeSnapshot,
    ) -> ResearchState:
        task = landscape.task
        exact_evidence = tuple(
            evidence for evidence in collection.evidence if _matches_task(evidence, task)
        )
        initial_baseline = _initial_baseline(collection, landscape, task)
        global_best_value, global_best_ids, incumbent = _global_best(
            exact_evidence,
            task,
        )
        executed = _executed_pipelines(exact_evidence)
        steps = tuple(
            evidence
            for evidence in exact_evidence
            if isinstance(evidence, ResearchStepEvidence)
        )
        lineage = _lineage_edges(collection, steps, task)
        stop_status = _stop_status(initial_baseline, runtime)
        return cls(
            task=task,
            initial_baseline=(
                None
                if initial_baseline is None
                else MappingProxyType(plain_json(initial_baseline))
            ),
            global_best_value=global_best_value,
            global_best_evidence_ids=global_best_ids,
            global_incumbent_evidence_id=incumbent,
            executed_pipelines=MappingProxyType(executed),
            research_step_count=len(steps),
            research_step_evidence_ids=tuple(sorted(step.evidence_id for step in steps)),
            lineage_edges=tuple(MappingProxyType(edge) for edge in lineage),
            runtime_budget=MappingProxyType(runtime.as_dict()),
            stop_status=MappingProxyType(stop_status),
        )

    def evidence_for_pipeline(
        self,
        collection: EvidenceCollection,
        pipeline: Mapping[str, Any],
    ) -> ComponentEvidence | ResearchStepEvidence | None:
        return collection.exact_pipeline_match(
            task={
                "dataset": self.task.dataset,
                "schema": self.task.schema,
                "unit": self.task.unit.as_dict(),
            },
            pipeline=pipeline,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.as_dict(),
            "initial_baseline": (
                None
                if self.initial_baseline is None
                else plain_json(self.initial_baseline)
            ),
            "global_best_value": self.global_best_value,
            "global_best_evidence_ids": list(self.global_best_evidence_ids),
            "global_incumbent_evidence_id": self.global_incumbent_evidence_id,
            "executed_pipelines": dict(self.executed_pipelines),
            "research_step_count": self.research_step_count,
            "research_step_evidence_ids": list(self.research_step_evidence_ids),
            "lineage_edges": [dict(edge) for edge in self.lineage_edges],
            "runtime_budget": dict(self.runtime_budget),
            "stop_status": dict(self.stop_status),
        }


def build_research_state(
    collection: EvidenceCollection,
    landscape: ComponentLandscape,
    runtime: ResearchRuntimeSnapshot,
) -> ResearchState:
    return ResearchState.from_evidence(collection, landscape, runtime)


def _matches_task(
    evidence: ComponentEvidence | ResearchStepEvidence,
    task: TaskContext,
) -> bool:
    return (
        evidence.task["dataset"] == task.dataset
        and evidence.task["schema"] == task.schema
        and plain_json(evidence.task["unit"]) == plain_json(task.unit.as_dict())
    )


def _initial_baseline(
    collection: EvidenceCollection,
    landscape: ComponentLandscape,
    task: TaskContext,
) -> dict[str, Any] | None:
    evidence_id = landscape.default_baseline_evidence_id
    if evidence_id is None:
        return None
    evidence = collection.by_id(evidence_id)
    if not isinstance(evidence, ComponentEvidence) or not _matches_task(evidence, task):
        raise ValueError("initial baseline must resolve to exact-task ComponentEvidence")
    return {
        "evidence_id": evidence.evidence_id,
        "pipeline": plain_json(evidence.pipeline),
        "target_value": _target_value(evidence, task),
    }


def _global_best(
    evidence: tuple[ComponentEvidence | ResearchStepEvidence, ...],
    task: TaskContext,
) -> tuple[float | None, tuple[str, ...], str | None]:
    if not evidence:
        return None, (), None
    values = {item.evidence_id: _target_value(item, task) for item in evidence}
    best = max(values.values())
    best_ids = tuple(
        sorted(evidence_id for evidence_id, value in values.items() if value == best)
    )
    return best, best_ids, best_ids[0]


def _target_value(
    evidence: ComponentEvidence | ResearchStepEvidence,
    task: TaskContext,
) -> float:
    value = evidence.metrics.get(task.target_metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evidence is missing target metric: {evidence.evidence_id}")
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"Evidence target metric is not finite: {evidence.evidence_id}")
    return metric


def _executed_pipelines(
    evidence: tuple[ComponentEvidence | ResearchStepEvidence, ...],
) -> dict[str, str]:
    executed: dict[str, str] = {}
    for item in sorted(evidence, key=lambda evidence_item: evidence_item.evidence_id):
        identity = pipeline_identity(item)
        existing = executed.get(identity)
        if existing is not None and existing != item.evidence_id:
            raise ValueError("exact task + normalized Pipeline matched multiple Evidence")
        executed[identity] = item.evidence_id
    return dict(sorted(executed.items()))


def _lineage_edges(
    collection: EvidenceCollection,
    steps: tuple[ResearchStepEvidence, ...],
    task: TaskContext,
) -> tuple[dict[str, str], ...]:
    edges = []
    for step in steps:
        parent = collection.by_id(step.baseline_evidence_id)
        if not _matches_task(parent, task):
            raise ValueError("ResearchStepEvidence baseline is outside the exact task")
        edges.append(
            {
                "parent_evidence_id": step.baseline_evidence_id,
                "child_evidence_id": step.evidence_id,
            }
        )
    return tuple(
        sorted(
            edges,
            key=lambda edge: (edge["parent_evidence_id"], edge["child_evidence_id"]),
        )
    )


def _stop_status(
    initial_baseline: Mapping[str, Any] | None,
    runtime: ResearchRuntimeSnapshot,
) -> dict[str, Any]:
    if initial_baseline is None:
        return {
            "can_continue": False,
            "stop_source": "harness",
            "stop_reason": "no_initial_baseline",
        }
    if runtime.model_finish_reason is not None:
        return {
            "can_continue": False,
            "stop_source": "model_finish",
            "model_finish_reason": runtime.model_finish_reason,
        }
    if runtime.experiments_remaining <= 0:
        return {
            "can_continue": False,
            "stop_source": "harness",
            "stop_reason": "experiment_budget_exhausted",
        }
    if (
        runtime.max_wall_seconds is not None
        and runtime.wall_seconds_used is not None
        and runtime.wall_seconds_used >= runtime.max_wall_seconds
    ):
        return {
            "can_continue": False,
            "stop_source": "harness",
            "stop_reason": "wall_time_exhausted",
        }
    return {
        "can_continue": True,
        "stop_source": None,
        "stop_reason": None,
    }


__all__ = [
    "ResearchRuntimeSnapshot",
    "ResearchState",
    "build_research_state",
]
