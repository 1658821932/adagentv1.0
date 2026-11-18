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

def process_identity(pid):
    try:
        # Linux start-time distinguishes a reused PID from the original worker.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None

def alive(manifest):
    pid = manifest.get("pid")
    return bool(pid and manifest.get("process_start") and
                process_identity(pid) == manifest["process_start"])


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
                        env={**os.environ, **trace_env, "PYTHONUNBUFFERED": "1"})
                manifest.update(pid=process.pid, process_start=process_identity(process.pid))
                write(folder / "manifest.json", manifest)
            index[key] = job_id
            write(index_path, index)
            return job_id

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
                children_path = folder / "children.json"
                with locked(folder / "children.lock"):
                    children = json.loads(children_path.read_text()) if children_path.exists() else []
                for child in children:
                    try:
                        os.killpg(child["pid"], 0)
                    except ProcessLookupError:
                        continue
                    return dict(status="running", tool=manifest["tool"],
                                job=dict(job_id=job_id, state="orphaned",
                                         message="Worker exited but a child process group remains; no replacement will be started.",
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
            records = [dict(pid=manifest.get("pid"), process_start=manifest.get("process_start"))]
            with locked(folder / "children.lock"):
                path = folder / "children.json"
                records.extend(json.loads(path.read_text()) if path.exists() else [])
            groups = []
            for record in records:
                pid = record.get("pid")
                if type(pid) is not int or pid <= 0:
                    continue
                try:
                    os.killpg(pid, 0)
                except (ProcessLookupError, PermissionError):
                    continue
                groups.append(pid)
            for pid in groups:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                live = []
                for record in records:
                    pid = record.get("pid")
                    if type(pid) is int and pid in groups and process_identity(pid) == record.get("process_start"):
                        live.append(pid)
                if not live:
                    break
                time.sleep(.05)
            for pid in groups:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
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
                self.stop(folder.name)
                stopped.append(folder.name)
        return stopped

    def wait(self, job_id, timeout):
        started = time.monotonic()
        while True:
            result = self.query(job_id)
            if result["status"] != "running" or (timeout is not None and time.monotonic()-started >= timeout):
                return result
            time.sleep(min(.2, max(.001, timeout-(time.monotonic()-started))) if timeout is not None else .2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stop-scope"])
    parser.add_argument("scope")
    args = parser.parse_args()
    print(json.dumps({"stopped": JobStore().stop_scope(args.scope)}))
