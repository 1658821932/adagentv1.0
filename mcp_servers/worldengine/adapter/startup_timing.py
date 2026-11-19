"""Small per-process wall-clock traces; no model inputs, weights or credentials."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import warnings


def _utc():
    return datetime.now(timezone.utc).isoformat()


class StartupTiming:
    def __init__(self, component, folder):
        self.component = component
        self.folder = Path(folder) if folder else None
        self.pid = None
        self._reset()

    def _reset(self):
        self.pid = os.getpid()
        self.started = time.perf_counter()
        launched = os.environ.get('ADAGENT_' + self.component.upper() + '_LAUNCHED_AT')
        self.record = dict(schema_version='native_startup_timing.v1', component=self.component,
            pid=self.pid, parent_pid=os.getppid(), started_at=_utc(), phases=[],
            component_launch_to_recorder_s=None if not launched else max(0., time.time() - float(launched)),
            launch_clock_scope='Component process group; workers can inherit a launch time preceding parent initialization.',
            measurement='Process wall time. Nested/parallel phases overlap; do not sum them.')
        self.active = {}
        self.warned = False

    def _write(self):
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / (self.component + '-' + str(self.pid) + '.json')
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self.record, ensure_ascii=False, indent=2))
            temporary.replace(path)
        except OSError:
            if not self.warned:
                warnings.warn('Cold-start timing could not be saved; simulation continues', RuntimeWarning)
                self.warned = True

    def begin(self, phase):
        if self.folder is None:
            return
        if self.pid != os.getpid():
            self._reset()
        index = len(self.record['phases'])
        self.active[phase] = (index, time.perf_counter())
        self.record['phases'].append(dict(phase=phase, status='running', started_at=_utc(),
            start_offset_s=round(time.perf_counter() - self.started, 6)))
        self._write()

    def end(self, phase, error=None):
        if self.folder is None or self.pid != os.getpid() or phase not in self.active:
            return
        index, started = self.active.pop(phase)
        record = self.record['phases'][index]
        record.update(status='failed' if error else 'completed', finished_at=_utc(),
            elapsed_seconds=round(time.perf_counter() - started, 6))
        if error:
            record['error_type'] = type(error).__name__
        self._write()

    def mark(self, phase):
        self.begin(phase)
        self.end(phase)

    @contextmanager
    def span(self, phase):
        self.begin(phase)
        try:
            yield
        except BaseException as error:
            self.end(phase, error)
            raise
        else:
            self.end(phase)


_TIMINGS = {}


def get_timing(component):
    folder = os.environ.get('ADAGENT_STARTUP_TIMING_DIR')
    key = (os.getpid(), component, folder)
    if key not in _TIMINGS:
        _TIMINGS[key] = StartupTiming(component, folder)
    return _TIMINGS[key]


def summarize(folder):
    folder = Path(folder)
    processes = []
    for path in sorted(folder.glob('*.json')):
        if path.name == 'summary.json':
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if value.get('schema_version') == 'native_startup_timing.v1':
            processes.append(value)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'summary.json'
    report = dict(schema_version='native_startup_summary.v1',
        status='recorded' if processes else 'not_recorded', processes=processes,
        measurement='Wall times include CPU/I/O waits; GPU work is not forcibly synchronized. Nested/parallel phases overlap.')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temporary.replace(path)
    return path
