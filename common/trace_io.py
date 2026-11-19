"""Execution trace context and redacted atomic JSON shared across adapters."""
from contextlib import contextmanager
import contextvars
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from uuid import uuid4


TRACE = contextvars.ContextVar('execution_trace', default={})


@contextmanager
def trace_context(**fields):
    token = TRACE.set({**TRACE.get(), **fields})
    try:
        yield
    finally:
        TRACE.reset(token)


def safe(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        sensitive = ('api_key', 'access_key', 'secret', 'password', 'credential', 'authorization')
        return {str(key): '[REDACTED]' if any(item in str(key).lower() for item in sensitive) else safe(child)
                for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)
