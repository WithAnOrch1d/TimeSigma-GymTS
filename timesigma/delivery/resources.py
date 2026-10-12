from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from .protocol import REPOSITORY


def bounded_process(command: list[str], log: Path, *, seconds: int, memory_mb: int, cpu_percent: int, cuda: bool = False, backend: str = "systemd", minimum_free_bytes: int | None = None) -> dict:
    if minimum_free_bytes is not None and (backend != "process_group" or type(minimum_free_bytes) is not int or minimum_free_bytes <= 0):
        raise ValueError("disk reserve requires the process_group monitor and a positive byte limit")
    environment = {name: "2" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
    environment.update(CUDA_VISIBLE_DEVICES="0" if cuda else "-1", PYTHONHASHSEED="1")
    limits = {"backend": backend, "memory_mb": memory_mb, "cpu_percent": cpu_percent, "wall_seconds": seconds, "device": "cuda:0" if cuda else "cpu"}
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("x") as output:
        if backend == "systemd":
            invocation = [
                "systemd-run", "--user", "--pipe", "--wait", "--collect", "--quiet",
                "-p", f"MemoryMax={memory_mb}M", "-p", "MemorySwapMax=1024M",
                "-p", f"CPUQuota={cpu_percent}%", "-p", "TasksMax=256",
                "-p", f"RuntimeMaxSec={seconds}s", "-p", "TimeoutStopSec=10s",
                *(f"--setenv={name}={value}" for name, value in environment.items()), "--", *command,
            ]
            result = subprocess.run(invocation, cwd=REPOSITORY, stdout=output, stderr=subprocess.STDOUT, timeout=seconds + 30)
            usage = {"returncode": result.returncode}
        elif backend == "process_group":
            usage = _run_process_group(command, output, seconds, memory_mb, cpu_percent, environment, minimum_free_bytes)
        else:
            raise ValueError("unknown declared resource backend; no unbounded fallback")
    return {**usage, "wall_seconds": time.monotonic() - started, "resource_limits": limits}


def _session_resources(session: int) -> tuple[int, int]:
    resident = 0
    processes = 0
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[3]) != session or stat[0] == "Z":
                continue
            resident += int((directory / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            processes += 1
        except (FileNotFoundError, ProcessLookupError):
            continue  # A child may exit while its procfs entry is being read.
    return resident, processes


def _kill_session(session: int) -> None:
    try:
        os.killpg(session, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_process_group(command, output, seconds, memory_mb, cpu_percent, environment, minimum_free_bytes=None) -> dict:
    if cpu_percent % 100:
        raise ValueError("process_group backend requires a whole number of CPU cores")
    cpus = sorted(os.sched_getaffinity(0))[:math.ceil(cpu_percent / 100)]
    if len(cpus) != cpu_percent // 100:
        raise ValueError("the declared CPU allocation is unavailable")
    invocation = ["taskset", "--cpu-list", ",".join(map(str, cpus)), *command]
    started = time.monotonic()
    process = subprocess.Popen(invocation, cwd=REPOSITORY, env={**os.environ, **environment}, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    peak = 0
    reason = None
    try:
        while process.poll() is None:
            resident, count = _session_resources(process.pid)
            peak = max(peak, resident)
            if time.monotonic() - started >= seconds:
                reason = "wall_time_exhausted"
            elif resident > memory_mb * 1024**2:
                reason = "rss_memory_exhausted"
            elif count > 256:
                reason = "process_count_exhausted"
            elif minimum_free_bytes is not None and shutil.disk_usage(Path(output.name).parent).free < minimum_free_bytes:
                reason = "disk_reserve_exhausted"
            if reason:
                _kill_session(process.pid)
                break
            time.sleep(0.1)
        process.wait(timeout=10)
    finally:
        _kill_session(process.pid)
        process.wait(timeout=10)
    return {
        "returncode": process.returncode, "limit_reason": reason, "peak_group_rss_bytes": peak,
        "cpu_affinity": cpus,
        "memory_enforcement": "100ms process-session RSS sampling and group kill; shared pages count repeatedly; not a kernel per-job memory cgroup",
        "minimum_free_bytes": minimum_free_bytes,
    }


def _nvidia_snapshot() -> dict:
    observations = {}
    for name, query in (("device", "--query-gpu=uuid,name,memory.total,memory.used,memory.free,driver_version"), ("processes", "--query-compute-apps=pid,process_name,used_gpu_memory")):
        result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits", "--id=0"], capture_output=True, text=True, timeout=10)
        observations[name] = {"returncode": result.returncode, "csv": result.stdout.strip(), "error": result.stderr.strip()}
    return observations


@contextmanager
def cuda_resource_usage(path: Path, memory_mb: int | None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the pinned AnomalyTransformer requires CUDA; no CPU adaptation is permitted")
    device = torch.cuda.get_device_properties(0)
    if memory_mb is not None:
        torch.cuda.set_per_process_memory_fraction(min(memory_mb * 1024**2 / device.total_memory, 1.0), 0)
    torch.cuda.reset_peak_memory_stats(0)
    observation = {
        "device": "cuda:0", "name": device.name, "total_memory_bytes": device.total_memory,
        "allocator_limit_mb": memory_mb, "allocator_backend": torch.cuda.get_allocator_backend(),
        "allocator_policy": "runtime_default" if memory_mb is None else "explicit_limit",
        "allocator_environment": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_runtime": torch.version.cuda, "torch": torch.__version__, "pid": os.getpid(),
        "before": _nvidia_snapshot(), "status": "failed",
    }
    started = time.monotonic()
    try:
        yield
        torch.cuda.synchronize(0)
        observation["status"] = "complete"
    finally:
        observation.update(
            wall_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            after=_nvidia_snapshot(),
            measurement_note="PyTorch allocator peaks cover this worker lifecycle; NVIDIA snapshots are device/process observations, not sampled peaks or per-job exclusivity guarantees.",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(observation, indent=2, allow_nan=False) + "\n")
