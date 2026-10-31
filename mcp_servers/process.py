"""Stream subprocess logs and clean up the complete job on failure/cancellation."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import threading


class ProcessCancelled(KeyboardInterrupt):
    """The current process received an explicit termination request."""


def stop_group(process, grace=2):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader may have exited while grandchildren are still alive.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(command, *, cwd, log_path, env=None, timeout=86400, input=None,
               json_stdout=False):
    """No pipe buffering: logs are visible while the process is running."""
    if type(timeout) not in (int, float) or timeout <= 0:
        raise ValueError("timeout must be positive")
    env = {**(os.environ if env is None else env), "PYTHONUNBUFFERED": "1"}
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path.with_suffix(".stdout.json") if json_stdout else log_path
    handler = None
    process = None
    try:
        if threading.current_thread() is threading.main_thread():
            handler = signal.getsignal(signal.SIGTERM)
            def terminated(signum, frame):
                raise ProcessCancelled("Job terminated")
            signal.signal(signal.SIGTERM, terminated)
        with log_path.open("w") as errors:
            errors.write("$ " + " ".join(map(str, command)) + "\n")
            errors.flush()
            if json_stdout:
                output = stdout_path.open("w")
            else:
                output = errors
            try:
                process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                    stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                    stdout=output, stderr=errors, start_new_session=True)
                try:
                    process.communicate(input=input, timeout=timeout)
                except BaseException:
                    stop_group(process)
                    raise
            finally:
                if output is not errors:
                    output.close()
        if process.returncode:
            stop_group(process)
        stdout = stdout_path.read_text(errors="replace") if json_stdout else _tail(stdout_path)
        stderr = _tail(log_path) if json_stdout else ""
        return subprocess.CompletedProcess(command, process.returncode,
                                           stdout if json_stdout else stdout[-16000:], stderr[-16000:])
    finally:
        if handler is not None:
            signal.signal(signal.SIGTERM, handler)


def _tail(path, size=65536):
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-size))
        return stream.read().decode("utf-8", errors="replace")
