from __future__ import annotations

import os
import sys
from typing import Mapping

from .protocol import ALLOWED_TOOLS, ToolRequestEnvelope


def dispatch_request(
    request: ToolRequestEnvelope, *, worker_pid: int | None = None
) -> dict[str, object]:
    if request.tool == "system.health":
        _exact_payload(request.payload, set(), request.tool)
        return {
            "protocol_version": request.protocol_version,
            "schema_version": request.schema_version,
            "allowed_tools": sorted(ALLOWED_TOOLS),
            "worker_pid": os.getpid() if worker_pid is None else worker_pid,
            "python_version": sys.version.split()[0],
        }

    if request.tool.startswith("delivery.") or "delivery_root" in request.payload:
        from timesigma.delivery.planning import dispatch_planning

        return dispatch_planning(request.tool, request.payload)

    from timesigma.research.tools import dispatch_research_tool

    return dispatch_research_tool(request.tool, request.payload)


def _exact_payload(
    payload: Mapping[str, object], expected: set[str], tool: str
) -> None:
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown:
        raise ValueError(f"Unknown {tool} payload field: {unknown[0]}")
    if missing:
        raise ValueError(f"Missing {tool} payload field: {missing[0]}")
