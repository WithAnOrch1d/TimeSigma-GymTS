from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Mapping


PROTOCOL_VERSION = "timesigma.bridge.v1"
SCHEMA_VERSION = 1
ALLOWED_TOOLS = frozenset(
    {
        "experiment.run",
        "experiment.preview",
        "memory.search",
        "memory.read",
        "memory.compare",
        "capability.lookup",
        "recommendation.context",
        "recommendation.evaluate",
        "recommendation.freeze",
        "research.complete_step",
        "research.context",
        "research.finish_episode",
        "system.health",
        "delivery.context",
        "delivery.preview",
        "delivery.freeze",
    }
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_SAFE_INTEGER = 2**53 - 1


class ToolStatus(str, Enum):
    SUCCESS = "success"
    METHOD_ERROR = "method_error"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings")
    return value


def _exact_fields(
    payload: Mapping[str, object], expected: set[str], contract: str
) -> None:
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown:
        raise ValueError(f"Unknown {contract} field: {unknown[0]}")
    if missing:
        raise ValueError(f"Missing {contract} field: {missing[0]}")


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return value


@dataclass(frozen=True)
class ToolRequestEnvelope:
    protocol_version: str
    schema_version: int
    request_id: str
    tool: str
    deadline_unix_ms: int
    idempotency_key: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        if not isinstance(self.schema_version, int) or isinstance(
            self.schema_version, bool
        ):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        _identifier(self.request_id, "request_id")
        if self.tool not in ALLOWED_TOOLS:
            raise ValueError(f"tool is not allowlisted: {self.tool}")
        if not isinstance(self.deadline_unix_ms, int) or isinstance(
            self.deadline_unix_ms, bool
        ):
            raise TypeError("deadline_unix_ms must be an integer")
        if self.deadline_unix_ms < 1:
            raise ValueError("deadline_unix_ms must be positive")
        if self.deadline_unix_ms > _MAX_SAFE_INTEGER:
            raise ValueError("deadline_unix_ms must be a safe integer")
        _identifier(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "payload", dict(_mapping(self.payload, "payload")))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "tool": self.tool,
            "deadline_unix_ms": self.deadline_unix_ms,
            "idempotency_key": self.idempotency_key,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ToolRequestEnvelope":
        payload = _mapping(payload, "ToolRequestEnvelope")
        fields = {
            "protocol_version",
            "schema_version",
            "request_id",
            "tool",
            "deadline_unix_ms",
            "idempotency_key",
            "payload",
        }
        _exact_fields(payload, fields, "ToolRequestEnvelope")
        return cls(**{field: payload[field] for field in fields})


@dataclass(frozen=True)
class CancelEnvelope:
    control: str
    request_id: str
    reason: str

    def __post_init__(self) -> None:
        if self.control != "cancel":
            raise ValueError("unsupported control message")
        _identifier(self.request_id, "request_id")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CancelEnvelope":
        payload = _mapping(payload, "CancelEnvelope")
        fields = {"control", "request_id", "reason"}
        _exact_fields(payload, fields, "CancelEnvelope")
        return cls(**{field: payload[field] for field in fields})


@dataclass(frozen=True)
class ToolResultEnvelope:
    request_id: str
    status: ToolStatus
    result: Mapping[str, object] | None
    error: Mapping[str, object] | None

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        if not isinstance(self.status, ToolStatus):
            raise TypeError("status must be ToolStatus")
        if (self.status is ToolStatus.SUCCESS) != (self.result is not None):
            raise ValueError("only successful results may contain result")
        if (self.status is ToolStatus.SUCCESS) == (self.error is not None):
            raise ValueError("failed results must contain error")
        if self.result is not None:
            object.__setattr__(self, "result", dict(_mapping(self.result, "result")))
        if self.error is not None:
            object.__setattr__(self, "error", dict(_mapping(self.error, "error")))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "status": self.status.value,
            "result": dict(self.result) if self.result is not None else None,
            "error": dict(self.error) if self.error is not None else None,
        }
