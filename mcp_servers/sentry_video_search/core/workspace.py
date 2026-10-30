"""Native standalone retrieval workspaces; adapters may bind an existing task."""
from __future__ import annotations
import contextvars
import functools
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_ACTIVE = contextvars.ContextVar("service_runs", default={})
ROOT = Path(__file__).resolve().parents[1]


def _safe(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if any(s in str(k).lower() for s in
                ("api_key", "access_key", "secret", "password", "credential", "authorization")) else _safe(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temp.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class Run:
    def __init__(self, service, operation, *, directory=None):
        self.service, self.operation = service, operation
        if directory is None:
            root = Path(os.environ.get("SENTRY_RUNTIME_ROOT", str(ROOT / "runtime"))).expanduser()
            directory = root / (datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_") + uuid4().hex[:12])
            directory.mkdir(parents=True, exist_ok=False)
        self.root = Path(directory).resolve()
        self.input = self.root / "input"
        self.logs = self.root / "logs"
        self.output = self.root / "output"
        for p in (self.input, self.logs, self.output):
            p.mkdir(parents=True, exist_ok=True)
        self.manifest = {"run_id": self.root.name, "service": service, "operation": operation,
                         "run_dir": str(self.root), "started_at": datetime.now(timezone.utc).isoformat(),
                         "status": "running"}
        if directory is not None and (self.root / "manifest.json").is_file():
            self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.update()

    def update(self, **fields):
        self.manifest.update(fields)
        write_json(self.root / "manifest.json", self.manifest)



def current_run(service):
    return _ACTIVE.get()[service]


def runtime_task(service, operation):
    """Allocate a task; nested calls and explicit child processes share it."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if service in _ACTIVE.get():
                return fn(*args, **kwargs)
            run = Run(service, operation)
            token = _ACTIVE.set({**_ACTIVE.get(), service: run})
            try:
                write_json(run.input / "request.json", {"args": args, "kwargs": kwargs})
                result = fn(*args, **kwargs)
                if isinstance(result, dict):
                    result.setdefault("run_dir", str(run.root))
                payload = _safe(result)
                if isinstance(payload, dict):
                    payload = {"status": "success", **payload}
                else:
                    payload = {"status": "success", "result": payload}
                write_json(run.output / "result.json", payload)
                run.update(status="success", finished_at=datetime.now(timezone.utc).isoformat())
                return result
            except BaseException as exc:
                run.update(status="error", error_type=type(exc).__name__, finished_at=datetime.now(timezone.utc).isoformat())
                raise
            finally:
                _ACTIVE.reset(token)
        return wrapped
    return decorate


from contextlib import contextmanager

@contextmanager
def bind_run(run):
    """Use a caller-supplied workspace object without importing its framework."""
    token = _ACTIVE.set({**_ACTIVE.get(), "sentry_video_search": run})
    try:
        yield
    finally:
        _ACTIVE.reset(token)
