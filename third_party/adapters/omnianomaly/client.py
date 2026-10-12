"""One local numeric subprocess; lifecycle and resource policy stay in the caller."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np

from .source import RUNTIME_DIRECTORY, PYTHON, UPSTREAMS, source_binding


class WorkerClient:
    def __init__(self, directory: Path, configuration, *, reference=False, seconds=3600):
        self.root = directory.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.log_path = self.root / "worker.log"
        self.log = self.log_path.open("xb")
        self.deadline = time.monotonic() + seconds
        self.process = None
        self.counter = 0
        try:
            sources = source_binding()
            if not PYTHON.is_file():
                raise RuntimeError("OmniAnomaly runtime missing; run scripts/setup_omnianomaly.py")
            environment = {**os.environ, **{k: "1" for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                           "CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
            self.process = subprocess.Popen([str(PYTHON), "-I", "-B", "-u", str(RUNTIME_DIRECTORY / "worker.py")],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                            env=environment, text=True, bufsize=1)
            self.runtime = self.call({"action": "initialize", "directory": str(self.root),
                                      "source_root": str(UPSTREAMS), "sources": sources,
                                      "configuration": configuration, "seed": 1, "reference": reference})["runtime"]
        except BaseException:
            self.close()
            raise

    def call(self, request):
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("OmniAnomaly worker exited; see " + str(self.log_path))
        try:
            process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            process.stdin.flush()
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                raise TimeoutError("OmniAnomaly worker exceeded the declared wall time")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("OmniAnomaly worker failed; see " + str(self.log_path))
            return json.loads(line)
        except BaseException:
            self.close()
            raise

    def numeric(self, values):
        from .binding import array_binding

        array = np.asarray(values)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != 1 or array.dtype.kind not in "fiu" or not len(array) or not np.isfinite(array).all():
            raise ValueError("OmniAnomaly requires finite scalar numeric observations")
        array = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("OmniAnomaly input is not finite after float32 conversion")
        name = f"input-{self.counter}.npy"
        self.counter += 1
        with (self.root / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        return {"path": name, **array_binding(array)}

    def score(self, record):
        from .binding import array_binding

        path = self.root / record["path"]
        if path.resolve().parent != self.root:
            raise ValueError("worker score path escapes its output directory")
        score = np.load(path, allow_pickle=False)
        if score.ndim != 1 or not len(score) or not np.isfinite(score).all() or array_binding(score) != {k: record[k] for k in ("shape", "dtype", "sha256")}:
            raise ValueError("worker score binding differs")
        return score

    def close(self):
        process = self.process
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        process.stdin.write('{"action":"close"}\n')
                        process.stdin.flush()
                    except BrokenPipeError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                process.stdin.close()
                process.stdout.close()
                self.process = None
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
