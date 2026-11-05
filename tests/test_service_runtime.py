"""Task isolation and adapter compatibility without loading GPU models."""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_servers.runtime import current_run, runtime_task


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAGENT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.delenv("ADAGENT_ACTIVE_SERVICE", raising=False)
    monkeypatch.delenv("ADAGENT_ACTIVE_RUN", raising=False)


def test_concurrent_tasks_never_share_workspace():
    @runtime_task("worldengine", "isolation")
    def task(number):
        run = current_run("worldengine")
        (run.output / "value").write_text(str(number))
        return str(run.root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        directories = list(pool.map(task, range(12)))
    assert len(set(directories)) == 12
    for i, root in enumerate(map(Path, directories)):
        assert (root / "output" / "value").read_text() == str(i)
        assert json.loads((root / "manifest.json").read_text())["status"] == "success"


def test_nested_calls_share_task_and_requests_redact_secrets():
    @runtime_task("sentry_video_search", "inner")
    def inner():
        return str(current_run("sentry_video_search").root)
    @runtime_task("sentry_video_search", "outer")
    def outer(payload):
        return inner()
    root = Path(outer({"api_key": "do-not-record", "checkpoint": "/weights/base.pth"}))
    request = (root / "input/request.json").read_text()
    assert "do-not-record" not in request
    assert "/weights/base.pth" in request
    assert json.loads((root / "manifest.json").read_text())["operation"] == "outer"


def test_failures_have_manifest_even_before_backend_launch(tmp_path):
    from mcp_servers.sentry_video_search.adapter.service import search_local
    from mcp_servers.worldengine.adapter.training import train_policy
    from mcp_servers.worldengine_failure_analysis.adapter.service import analyze_scene
    cases = [
        ("sentry_video_search", lambda: search_local("")),
        ("worldengine", lambda: train_policy({"training_mode": "unknown"})),
        ("worldengine_failure_analysis", lambda: analyze_scene("")),
    ]
    for service, call in cases:
        with pytest.raises(Exception):
            call()
        manifests = list((tmp_path / "runtime" / service).glob("*/manifest.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text())["status"] == "error"
