from __future__ import annotations

# noqa: SIZE_OK - JSONL process supervision is one transport responsibility.

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import multiprocessing
from multiprocessing.connection import Connection
import os
import signal
import sys
import threading
import time
from typing import Mapping

from .dispatch import dispatch_request
from .protocol import (
    CancelEnvelope,
    ToolRequestEnvelope,
    ToolResultEnvelope,
    ToolStatus,
)


_COMPLETED_CACHE_LIMIT = 1024


@dataclass
class _Job:
    request: ToolRequestEnvelope
    identity: str
    process: multiprocessing.Process
    receive: Connection
    cancelled_reason: str | None = None
    deadline_exceeded: bool = False


class WorkerServer:
    def __init__(
        self,
        *,
        max_inflight: int = 4,
        dispatch: Callable[..., dict[str, object]] = dispatch_request,
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        self.max_inflight = max_inflight
        self._dispatch = dispatch
        self._context = multiprocessing.get_context("spawn")
        self._jobs: dict[str, _Job] = {}
        self._idempotency: dict[str, tuple[str, dict[str, object]]] = {}
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._closing = False

    def serve(self) -> None:
        for line in sys.stdin:
            if self._closing:
                break
            if not line.strip():
                continue
            frame: object = None
            try:
                frame = json.loads(line)
                if not isinstance(frame, Mapping):
                    raise TypeError("JSONL frame must be an object")
                if frame.get("control") == "cancel":
                    self._cancel(CancelEnvelope.from_dict(frame))
                else:
                    self._submit(ToolRequestEnvelope.from_dict(frame))
            except (TypeError, ValueError) as exc:
                request_id = _safe_request_id(frame)
                if request_id is None:
                    print(
                        f"invalid JSONL frame: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                else:
                    self._emit_error(
                        request_id,
                        ToolStatus.METHOD_ERROR,
                        "INVALID_ENVELOPE",
                        exc,
                    )
        self.shutdown()

    def _submit(self, request: ToolRequestEnvelope) -> None:
        if request.deadline_unix_ms <= int(time.time() * 1000):
            self._emit_error(
                request.request_id,
                ToolStatus.CANCELLED,
                "DEADLINE_EXCEEDED",
                TimeoutError("request deadline expired before dispatch"),
            )
            return

        identity = _request_identity(request)
        with self._lock:
            previous = self._idempotency.get(request.idempotency_key)
            if previous is not None:
                previous_identity, response = previous
                if previous_identity != identity:
                    self._emit_error(
                        request.request_id,
                        ToolStatus.METHOD_ERROR,
                        "IDEMPOTENCY_CONFLICT",
                        ValueError("idempotency key was reused for another request"),
                    )
                else:
                    replay = dict(response)
                    replay["request_id"] = request.request_id
                    self._emit(replay)
                return
            inflight = next(
                (
                    job
                    for job in self._jobs.values()
                    if job.request.idempotency_key == request.idempotency_key
                ),
                None,
            )
            if inflight is not None:
                if inflight.identity != identity:
                    code = "IDEMPOTENCY_CONFLICT"
                    message = "idempotency key was reused for another request"
                else:
                    code = "IDEMPOTENCY_IN_FLIGHT"
                    message = "idempotent request is already in flight"
                self._emit_error(
                    request.request_id,
                    ToolStatus.METHOD_ERROR,
                    code,
                    ValueError(message),
                )
                return
            if request.request_id in self._jobs:
                self._emit_error(
                    request.request_id,
                    ToolStatus.METHOD_ERROR,
                    "DUPLICATE_REQUEST_ID",
                    ValueError("request_id is already in flight"),
                )
                return
            if len(self._jobs) >= self.max_inflight:
                self._emit_error(
                    request.request_id,
                    ToolStatus.INFRA_ERROR,
                    "WORKER_BUSY",
                    RuntimeError("worker backpressure limit reached"),
                )
                return

            receive, send = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=_job_entry,
                args=(request.to_dict(), os.getpid(), self._dispatch, send),
            )
            self._jobs[request.request_id] = _Job(
                request, identity, process, receive
            )
            process.start()
            send.close()
            threading.Thread(
                target=self._monitor,
                args=(request.request_id,),
                daemon=True,
            ).start()

    def _monitor(self, request_id: str) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
        if job is None:
            return
        remaining_seconds = max(
            0.0, (job.request.deadline_unix_ms - int(time.time() * 1000)) / 1000
        )
        job.process.join(timeout=remaining_seconds)
        if job.process.is_alive():
            with self._lock:
                if job.cancelled_reason is None:
                    job.deadline_exceeded = True
            _terminate_job_process(job)

        with self._lock:
            if job.deadline_exceeded:
                response = _error(
                    request_id,
                    ToolStatus.CANCELLED,
                    "DEADLINE_EXCEEDED",
                    TimeoutError("request deadline expired during execution"),
                )
            elif job.cancelled_reason is not None:
                response = _error(
                    request_id,
                    ToolStatus.CANCELLED,
                    "CANCELLED",
                    RuntimeError(job.cancelled_reason),
                )
            elif job.process.exitcode != 0:
                response = _error(
                    request_id,
                    ToolStatus.INFRA_ERROR,
                    "JOB_PROCESS_EXITED",
                    RuntimeError(f"tool job exited with code {job.process.exitcode}"),
                )
            elif not job.receive.poll():
                response = _error(
                    request_id,
                    ToolStatus.INFRA_ERROR,
                    "MISSING_JOB_RESULT",
                    RuntimeError("tool job exited without a result"),
                )
            else:
                response = job.receive.recv()
            job.receive.close()
            self._jobs.pop(request_id, None)
            self._cache_completed(job.request.idempotency_key, job.identity, response)
        self._emit(response)

    def _cancel(self, cancel: CancelEnvelope) -> None:
        with self._lock:
            job = self._jobs.get(cancel.request_id)
            if job is None or not job.process.is_alive():
                return
            job.cancelled_reason = cancel.reason
        _terminate_job_process(job)

    def _cache_completed(
        self, key: str, identity: str, response: dict[str, object]
    ) -> None:
        self._idempotency[key] = (identity, response)
        if len(self._idempotency) > _COMPLETED_CACHE_LIMIT:
            self._idempotency.pop(next(iter(self._idempotency)))

    def _emit(self, payload: Mapping[str, object]) -> None:
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._write_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _emit_error(
        self,
        request_id: str,
        status: ToolStatus,
        code: str,
        error: BaseException,
    ) -> None:
        self._emit(_error(request_id, status, code, error))

    def shutdown(self) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.process.is_alive():
                _terminate_job_process(job)
        for job in jobs:
            job.process.join(timeout=2.0)


def _job_entry(
    request_payload: Mapping[str, object],
    worker_pid: int,
    dispatch: Callable[..., dict[str, object]],
    send: Connection,
) -> None:
    try:
        _isolate_job_process()
        os.dup2(2, 1)
        request = ToolRequestEnvelope.from_dict(request_payload)
        response = ToolResultEnvelope(
            request_id=request.request_id,
            status=ToolStatus.SUCCESS,
            result=dispatch(request, worker_pid=worker_pid),
            error=None,
        ).to_dict()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        response = _error(
            str(request_payload.get("request_id", "unknown")),
            ToolStatus.INFRA_ERROR,
            "IO_ERROR",
            exc,
        )
    except BaseException as exc:
        response = _error(
            str(request_payload.get("request_id", "unknown")),
            ToolStatus.METHOD_ERROR,
            "TOOL_ERROR",
            exc,
        )
    try:
        send.send(response)
    finally:
        send.close()


def _isolate_job_process() -> None:
    if not hasattr(os, "setsid"):
        return
    try:
        os.setsid()
    except OSError:
        return


def _terminate_job_process(job: _Job, *, timeout_seconds: float = 2.0) -> None:
    pid = job.process.pid
    process_group_id: int | None = None
    if pid is not None and hasattr(os, "getpgid"):
        try:
            process_group_id = os.getpgid(pid)
        except ProcessLookupError:
            process_group_id = None
    if _can_signal_process_group(process_group_id):
        assert process_group_id is not None
        _signal_process_group(process_group_id, signal.SIGTERM)
    else:
        job.process.terminate()
    job.process.join(timeout=timeout_seconds)
    if job.process.is_alive():
        if _can_signal_process_group(process_group_id):
            assert process_group_id is not None
            _signal_process_group(process_group_id, signal.SIGKILL)
        else:
            job.process.kill()
        job.process.join(timeout=timeout_seconds)


def _can_signal_process_group(process_group_id: int | None) -> bool:
    return (
        process_group_id is not None
        and hasattr(os, "killpg")
        and process_group_id != os.getpgrp()
    )


def _signal_process_group(process_group_id: int, sig: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return


def _error(
    request_id: str, status: ToolStatus, code: str, error: BaseException
) -> dict[str, object]:
    return ToolResultEnvelope(
        request_id=request_id,
        status=status,
        result=None,
        error={
            "code": code,
            "type": type(error).__name__,
            "message": str(error)[:1000],
        },
    ).to_dict()


def _request_identity(request: ToolRequestEnvelope) -> str:
    return json.dumps(
        {"tool": request.tool, "payload": request.payload},
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_request_id(payload: object) -> str | None:
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("request_id"), str
    ):
        return None
    try:
        return CancelEnvelope(
            control="cancel",
            request_id=payload["request_id"],
            reason="validation",
        ).request_id
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-inflight", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_inflight < 1:
        parser.error("--max-inflight must be positive")
    server = WorkerServer(max_inflight=args.max_inflight)
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
