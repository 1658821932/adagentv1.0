"""Persistent, process-isolated tool jobs. Waiting never cancels execution."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def write(path, data):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.chmod(temporary, 0o600)
    temporary.replace(path)

@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield

@dataclass(frozen=True)
class ProcessState:
    pid: int
    session: int
    start: str
    live: bool


def _process_state(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return ProcessState(pid, int(fields[3]), fields[19], fields[0] not in {'Z', 'X', 'x'})
    except (OSError, IndexError, ValueError):
        return None


def process_identity(pid):
    state = _process_state(pid)
    return state.start if state and state.live else None


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_record(pid):
    return dict(pid=pid, process_start=process_identity(pid), boot_id=_boot_id())


def alive(manifest):
    pid = manifest.get("pid")
    return bool(type(pid) is int and pid > 0 and manifest.get("process_start") and
                ('boot_id' not in manifest or manifest['boot_id'] == _boot_id()) and
                process_identity(pid) == manifest["process_start"])


def _has_job_marker(process, marker):
    try:
        values = Path(f'/proc/{process.pid}/environ').read_bytes().split(b'\0')
        return marker in values and process_identity(process.pid) == process.start
    except OSError:
        return False


def _session_members(records, job_id, known):
    """Return observed members and the subset backed by a live ownership anchor."""
    states = {}
    sessions = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal() and (state := _process_state(int(entry.name))) is not None:
            states[state.pid] = state
            if state.live:
                sessions.setdefault(state.session, []).append(state)
    boot = _boot_id()
    marker = f'ADAGENT_JOB_ID={job_id}'.encode()
    observed, verified = set(), set()
    for record in records:
        leader_pid = record.get('pid')
        start = str(record.get('process_start') or '')
        if type(leader_pid) is not int or leader_pid <= 0 or not start.isdecimal() or record.get('boot_id', boot) != boot:
            continue
        leader = states.get(leader_pid)
        if leader is not None and (leader.session != leader_pid or leader.start != start):
            continue
        members = {state for state in sessions.get(leader_pid, ()) if int(state.start) >= int(start)}
        observed.update(members)
        current_leader = _process_state(leader_pid) if leader is not None else None
        anchored = current_leader is not None and current_leader.start == start and current_leader.session == leader_pid
        if anchored or members & known or any(_has_job_marker(state, marker) for state in members):
            verified.update(members)
    known.update(verified)
    return observed, verified


def _signal_member(member, sig):
    """A pidfd addresses the same process even if the numeric PID is reused."""
    try:
        fd = os.pidfd_open(member.pid)
        try:
            if _process_state(member.pid) == member:
                signal.pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)
    except (ProcessLookupError, PermissionError):
        pass


def terminate_sessions(records, job_id, grace=2):
    """Terminate owned members and return any observed PID still present."""
    known = set()
    for sig, seconds in ((signal.SIGTERM, grace), (signal.SIGKILL, 2)):
        deadline = time.monotonic() + seconds
        signalled = set()
        while True:
            remaining, owned = _session_members(records(), job_id, known)
            if not remaining or time.monotonic() >= deadline:
                break
            for member in owned - signalled:
                _signal_member(member, sig)
            signalled.update(owned)
            time.sleep(.05)
    return sorted(state.pid for state in remaining)


def preparation_run_dir(manifest):
    if manifest.get('tool') != 'ego_trajectory_editor':
        return None
    root = PROJECT_ROOT/'mcp_servers/scene_editing/runtime'
    for path in sorted(root.glob('run_*/manifest.json'), key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get('operation') == 'ego_editing' and record.get('owner_pid') == manifest.get('pid'):
            return record.get('run_dir') or str(path.parent)
    return None


class JobStore:
    def __init__(self, root=None):
        self.root = Path(root or PROJECT_ROOT / "logs" / "jobs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _folder(self, job_id):
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid job_id")
        folder = self.root / job_id
        if not (folder / "manifest.json").is_file():
            raise ValueError("Unknown job_id")
        return folder

    def lookup(self, tool, mode, arguments, scope):
        """Find an already submitted call without spawning or retrying it."""
        signature = dict(tool=tool, mode=mode, arguments=arguments, scope=scope)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with locked(self.root / 'registry.lock'):
            index_path = self.root / 'index.json'
            index = json.loads(index_path.read_text()) if index_path.exists() else {}
            job_id = index.get(key)
            if job_id:
                self._folder(job_id)
            return job_id

    def submit(self, tool, mode, arguments, scope, *, retry=False):
        # References that change on disk must not silently match an old request.
        def refs(value):
            if isinstance(value, dict):
                return {k: refs(v) for k, v in value.items()}
            if isinstance(value, list):
                return [refs(v) for v in value]
            if isinstance(value, str):
                try:
                    p = Path(value)
                    if p.is_absolute() and p.exists():
                        st = p.stat()
                        return [value, st.st_size, st.st_mtime_ns]
                except OSError:
                    pass
            return value
        signature = dict(tool=tool, mode=mode, arguments=arguments, scope=scope)
        references = refs(arguments)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with locked(self.root / "registry.lock"):
            index_path = self.root / "index.json"
            index = json.loads(index_path.read_text()) if index_path.exists() else {}
            previous = index.get(key)
            if previous:
                state = self.query(previous)
                # retry cannot spawn a second copy of a still-running job.
                if state["status"] == "running":
                    return previous
                previous_manifest = json.loads((self._folder(previous) / "manifest.json").read_text())
                if not retry and previous_manifest.get("references") == references:
                    return previous
            job_id = uuid4().hex
            folder = self.root / job_id
            folder.mkdir(mode=0o700)
            write(folder / "request.json", dict(tool=tool, mode=mode, arguments=arguments))
            manifest = dict(job_id=job_id, tool=tool, mode=mode, scope=scope,
                            status="queued", started_at=time.time(), key=key, references=references)
            write(folder / "manifest.json", manifest)
            # Worker cannot claim the manifest until parent records its identity.
            with locked(folder / "state.lock"):
                from mcp_servers.runtime import _TRACE
                trace_env = {"ADAGENT_" + k.upper(): str(v) for k, v in _TRACE.get().items()}
                with (folder / "worker.log").open("w") as log:
                    os.chmod(folder / "worker.log", 0o600)
                    process = subprocess.Popen(
                        [sys.executable, "-m", "agent.job_worker", str(folder)],
                        cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, **trace_env, "ADAGENT_JOB_ID": job_id, "PYTHONUNBUFFERED": "1"})
                manifest.update(process_record(process.pid))
                write(folder / "manifest.json", manifest)
            index[key] = job_id
            write(index_path, index)
            return job_id

    def _records(self, folder, manifest):
        path = folder / 'children.json'
        with locked(folder / 'children.lock'):
            return [manifest, *(json.loads(path.read_text()) if path.exists() else [])]

    def query(self, job_id):
        folder = self._folder(job_id)
        with locked(folder / "state.lock"):
            manifest = json.loads((folder / "manifest.json").read_text())
            result_file = folder / "result.json"
            if result_file.exists():
                return {**json.loads(result_file.read_text()),
                        "job": dict(job_id=job_id, state=manifest["status"], manifest_path=str(folder / "manifest.json"),
                                    elapsed_seconds=(round(manifest['finished_at']-manifest['started_at'],3)
                                        if manifest.get('finished_at') is not None else None))}
            if not alive(manifest):
                # The worker may have died while its external processes survive.
                remaining, _ = _session_members(self._records(folder, manifest), job_id, set())
                if remaining:
                    return dict(status="running", tool=manifest["tool"],
                                job=dict(job_id=job_id, state="orphaned",
                                         message="Worker exited but descendants remain; no replacement will be started.",
                                         manifest_path=str(folder / "manifest.json")))
                manifest.update(status="interrupted", finished_at=time.time())
                write(folder / "manifest.json", manifest)
                recovery = preparation_run_dir(manifest)
                error = dict(type="job_interrupted", message="Worker exited without a result; inspect worker.log before explicit retry.")
                if recovery:
                    error['reuse_completed_from'] = recovery
                return dict(status="error", tool=manifest["tool"], error=error,
                            job=dict(job_id=job_id, state="interrupted"))
            return dict(status="running", tool=manifest["tool"],
                        job=dict(job_id=job_id, state=manifest["status"],
                                 elapsed_seconds=round(time.time()-manifest["started_at"], 2),
                                 manifest_path=str(folder / "manifest.json")))

    def stop(self, job_id):
        folder = self._folder(job_id)
        with locked(folder / "state.lock"):
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest.get("status") not in {"queued", "running"}:
                return manifest
            remaining = terminate_sessions(lambda: self._records(folder, manifest), job_id)
            if remaining:
                manifest['cleanup_pending_pids'] = remaining
                write(folder / 'manifest.json', manifest)
                return manifest
            manifest.pop('cleanup_pending_pids', None)
            result = dict(status="error", tool=manifest["tool"],
                          error=dict(type="job_interrupted", message="Job stopped by user"))
            manifest.update(status="interrupted", finished_at=time.time())
            write(folder / "result.json", result)
            write(folder / "manifest.json", manifest)
            return manifest

    def stop_scope(self, scope):
        stopped = []
        for folder in self.root.iterdir():
            if not folder.is_dir():
                continue
            try:
                manifest = json.loads((folder / "manifest.json").read_text())
            except (OSError, ValueError):
                continue
            if manifest.get("scope") == scope and manifest.get("status") in {"queued", "running"}:
                if self.stop(folder.name)['status'] == 'interrupted':
                    stopped.append(folder.name)
        return stopped

    def wait(self, job_id, timeout):
        started = time.monotonic()
        while True:
            result = self.query(job_id)
            if (result["status"] != "running" or (result.get('job') or {}).get('state') == 'orphaned'
                    or (timeout is not None and time.monotonic()-started >= timeout)):
                return result
            time.sleep(min(.2, max(.001, timeout-(time.monotonic()-started))) if timeout is not None else .2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stop-scope"])
    parser.add_argument("scope")
    args = parser.parse_args()
    print(json.dumps({"stopped": JobStore().stop_scope(args.scope)}))
