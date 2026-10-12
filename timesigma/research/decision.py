from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .pipeline import NextAction, _thaw


DIAGNOSIS_VERDICTS = ("supported", "refuted", "inconclusive")
EXECUTABLE_ACTION_DIMENSIONS = (
    "algorithm",
    "preprocess",
    "diff_order",
    "window",
    "hyperparameter",
    "preprocessing",
    "ensemble",
    "pipeline",
)
STOP_ACTION_DIMENSION = "stop"
STOP_ACTION_REASONS = ("target_saturated", "no_reasonable_next_step")


@dataclass(frozen=True)
class StopAction:
    value: str
    parameter: None = None
    dimension: str = STOP_ACTION_DIMENSION

    def __post_init__(self) -> None:
        if self.dimension != STOP_ACTION_DIMENSION:
            raise ValueError("stop action must use the stop dimension")
        if self.parameter is not None:
            raise ValueError("stop action does not accept a parameter name")
        if self.value not in STOP_ACTION_REASONS:
            raise ValueError(f"unsupported stop reason: {self.value}")

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "parameter": self.parameter,
            "value": self.value,
        }


ResearchNextAction = NextAction | StopAction


@dataclass(frozen=True)
class ResearchDecision:
    hypothesis: str
    diagnosis_verdict: str
    diagnosis_reason: str
    next_action: ResearchNextAction
    next_action_rationale: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("hypothesis", self.hypothesis),
            ("diagnosis_reason", self.diagnosis_reason),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")
        if (
            self.next_action_rationale is not None
            and not self.next_action_rationale.strip()
        ):
            raise ValueError("next_action_rationale must be non-empty when provided")
        if self.diagnosis_verdict not in DIAGNOSIS_VERDICTS:
            raise ValueError(
                f"unsupported diagnosis verdict: {self.diagnosis_verdict}"
            )
        if is_stop_action(self.next_action):
            return
        if self.next_action.dimension not in EXECUTABLE_ACTION_DIMENSIONS:
            raise ValueError("next_action must generate an executable Pipeline")

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "hypothesis": self.hypothesis,
            "diagnosis": {
                "verdict": self.diagnosis_verdict,
                "reason": self.diagnosis_reason,
            },
            "next_action": self.next_action.as_dict(),
        }
        if self.next_action_rationale is not None:
            payload["next_action_rationale"] = self.next_action_rationale
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ResearchDecision:
        legacy_fields = {"hypothesis", "diagnosis", "next_action"}
        current_fields = legacy_fields | {"next_action_rationale"}
        if set(payload) not in (legacy_fields, current_fields):
            _exact_fields(payload, current_fields, "research decision")
        diagnosis = _mapping(payload, "diagnosis")
        _exact_fields(diagnosis, {"verdict", "reason"}, "diagnosis")
        rationale = payload.get("next_action_rationale")
        if rationale is not None and not isinstance(rationale, str):
            raise TypeError("next_action_rationale must be a string")
        return cls(
            hypothesis=_string(payload, "hypothesis"),
            diagnosis_verdict=_string(diagnosis, "verdict"),
            diagnosis_reason=_string(diagnosis, "reason"),
            next_action=research_next_action_from_mapping(payload, "next_action"),
            next_action_rationale=rationale,
        )


def is_stop_action(action: ResearchNextAction) -> bool:
    return isinstance(action, StopAction)


def research_next_action_from_mapping(
    payload: Mapping[str, object],
    key: str,
) -> ResearchNextAction:
    action = _mapping(payload, key)
    _exact_fields(action, {"dimension", "parameter", "value"}, key)
    dimension = _string(action, "dimension")
    if dimension == STOP_ACTION_DIMENSION:
        parameter = action.get("parameter")
        if parameter is not None:
            raise ValueError(f"{key}.parameter must be null for stop")
        return StopAction(value=_string(action, "value"))
    return _action(payload, key)


def _action(payload: Mapping[str, object], key: str) -> NextAction:
    action = _mapping(payload, key)
    _exact_fields(action, {"dimension", "parameter", "value"}, key)
    dimension = _string(action, "dimension")
    parameter = action.get("parameter")
    if parameter is not None and not isinstance(parameter, str):
        raise TypeError(f"{key}.parameter must be a string or null")
    return NextAction(
        dimension=dimension,  # type: ignore[arg-type]
        parameter=parameter,
        value=_thaw(action.get("value")),
    )


def _exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    contract: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{contract} fields do not match the frozen contract")


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(payload.get(key), key)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = [
    "DIAGNOSIS_VERDICTS",
    "EXECUTABLE_ACTION_DIMENSIONS",
    "ResearchDecision",
    "ResearchNextAction",
    "STOP_ACTION_DIMENSION",
    "STOP_ACTION_REASONS",
    "StopAction",
    "is_stop_action",
    "research_next_action_from_mapping",
]
