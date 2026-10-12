from __future__ import annotations

import sys
import time

import pytest

from timesigma.bridge.dispatch import dispatch_request
from timesigma.bridge.protocol import (
    ALLOWED_TOOLS,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    CancelEnvelope,
    ToolRequestEnvelope,
    ToolResultEnvelope,
    ToolStatus,
)


def _request(
    request_id: str,
    *,
    payload: dict[str, object] | None = None,
) -> ToolRequestEnvelope:
    return ToolRequestEnvelope(
        protocol_version=PROTOCOL_VERSION,
        schema_version=SCHEMA_VERSION,
        request_id=request_id,
        tool="system.health",
        deadline_unix_ms=int(time.time() * 1000) + 30_000,
        idempotency_key=f"idem-{request_id}",
        payload={} if payload is None else payload,
    )


def test_request_and_result_envelopes_match_the_neutral_protocol() -> None:
    request = _request("request-round-trip")
    assert ToolRequestEnvelope.from_dict(request.to_dict()) == request
    assert set(request.to_dict()) == {
        "protocol_version",
        "schema_version",
        "request_id",
        "tool",
        "deadline_unix_ms",
        "idempotency_key",
        "payload",
    }
    with pytest.raises(ValueError, match="Unknown ToolRequestEnvelope field: track"):
        ToolRequestEnvelope.from_dict({**request.to_dict(), "track": "t0"})
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        ToolRequestEnvelope.from_dict({**request.to_dict(), "schema_version": True})
    with pytest.raises(TypeError, match="deadline_unix_ms must be an integer"):
        ToolRequestEnvelope.from_dict({**request.to_dict(), "deadline_unix_ms": True})
    with pytest.raises(ValueError, match="deadline_unix_ms must be a safe integer"):
        ToolRequestEnvelope.from_dict(
            {**request.to_dict(), "deadline_unix_ms": 2**53}
        )

    result = ToolResultEnvelope(
        request_id=request.request_id,
        status=ToolStatus.SUCCESS,
        result={"ready": True},
        error=None,
    ).to_dict()
    assert set(result) == {
        "protocol_version",
        "schema_version",
        "request_id",
        "status",
        "result",
        "error",
    }


def test_cancel_envelope_is_a_separate_control_message() -> None:
    cancel = CancelEnvelope.from_dict(
        {
            "control": "cancel",
            "request_id": "request-cancel",
            "reason": "caller stopped",
        }
    )
    assert cancel.reason == "caller stopped"
    with pytest.raises(ValueError, match="Unknown CancelEnvelope field"):
        CancelEnvelope.from_dict(
            {
                "protocol_version": PROTOCOL_VERSION,
                "control": "cancel",
                "request_id": "request-cancel",
                "reason": "caller stopped",
            }
        )


def test_dispatch_exposes_only_neutral_health_information() -> None:
    health = dispatch_request(_request("request-health"), worker_pid=1234)
    assert health == {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "allowed_tools": sorted(ALLOWED_TOOLS),
        "worker_pid": 1234,
        "python_version": sys.version.split()[0],
    }
    with pytest.raises(ValueError, match="Unknown system.health payload field"):
        dispatch_request(_request("request-invalid-health", payload={"track": "t0"}))
