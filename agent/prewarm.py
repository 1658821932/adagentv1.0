"""任务开始时预热重工具环境。

背景：视频检索每次都在独立子进程里跑，其中 ``import chromadb`` 在页缓存冷时
需要约 18s。这里在任务一开始就点火一个独立进程把它读热，使这笔开销与 LLM
首轮决策（通常 5~15s）重叠，而不是等工具真正调用时才付。

设计要点：
- 完全 fire-and-forget，不阻塞、不抛异常、不影响主流程；
- 故意不 import 检索组件，避免把重依赖带进 agent 进程；
- 幂等：重复调用只点火一次。
"""

from __future__ import annotations

import atexit
import os
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_RETRIEVER_PYTHON = PROJECT_ROOT / "environments/video_retriever/.venv/bin/python"

_lock = threading.Lock()
_proc: "subprocess.Popen | None" = None
_started = False


def _spawn() -> None:
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return
        if not VIDEO_RETRIEVER_PYTHON.is_file():
            return
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONUNBUFFERED": "1"}
        try:
            _proc = subprocess.Popen(
                [
                    str(VIDEO_RETRIEVER_PYTHON),
                    "-m",
                    "mcp_servers.sentry_video_search.adapter.prewarm",
                ],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            _proc = None


def start() -> None:
    """幂等点火，立即返回。"""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_spawn, name="video-retriever-prewarm", daemon=True).start()
    atexit.register(_stop)


def _stop() -> None:
    proc = _proc
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
