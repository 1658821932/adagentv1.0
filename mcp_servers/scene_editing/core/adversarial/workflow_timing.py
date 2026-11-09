"""Persist nested wall-clock spans without recording tool arguments or credentials."""
import contextvars
import functools
import inspect
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from experiment_runtime import atomic_json

_CURRENT = contextvars.ContextVar('workflow_timing_parent', default=None)
_ROOT = contextvars.ContextVar('workflow_timing_root', default=None)


def timed_operation(name, output_argument=None):
    """Time a stage/tool. CLI main spans use --run or --output as their root."""
    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            owner = bound.get('self', bound.get('editor'))
            root = bound.get(output_argument) if output_argument else getattr(owner, 'run_dir', None)
            if root is None and function.__name__ == 'main':
                for flag in ('--run', '--output'):
                    if flag in sys.argv:
                        index = sys.argv.index(flag)
                        if index+1 < len(sys.argv):
                            root = sys.argv[index+1]
                            break
            if root is None:
                root = _ROOT.get()
            if root is None:
                return function(*args, **kwargs)
            root = Path(root).resolve()
            # Do not create the output root before entry points validate it.
            event_id = uuid.uuid4().hex
            record = dict(schema_version='workflow_timing.v1', event_id=event_id,
                parent_event_id=_CURRENT.get(), name=name,
                started_at=datetime.now(timezone.utc).isoformat(), status='running')
            token = _CURRENT.set(event_id)
            root_token = _ROOT.set(root)
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                record['status'] = 'returned'
                if isinstance(result, dict) and isinstance(result.get('status'), str):
                    record['result_status'] = result['status']
                return result
            except BaseException as error:
                record.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                              error_type=type(error).__name__)
                raise
            finally:
                record.update(elapsed_seconds=round(time.perf_counter()-started, 6),
                              finished_at=datetime.now(timezone.utc).isoformat())
                _CURRENT.reset(token)
                _ROOT.reset(root_token)
                if root.is_dir():
                    atomic_json(root/'timing'/f'{event_id}.json', record)
        return wrapped
    return decorate
