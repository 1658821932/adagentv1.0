"""Protect the core/adapter boundary and standalone command line entry points."""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ["dggt", "smart_edit", "sentry_video_search", "worldengine", "worldengine_failure_analysis"]


@pytest.mark.parametrize("service", SERVICES)
def test_consistent_layout_and_no_framework_import_in_core(service):
    root = ROOT / "mcp_servers" / service
    for name in ("core", "adapter"):
        assert (root / name).is_dir()
    for name in ("service.py", "config.py", "schemas.py", "tests"):
        assert (root / "adapter" / name).exists()
    assert (root / "README.md").is_file()
    assert (root / "core/README.md").is_file()
    excluded = {"__pycache__", "data", "pretrained", "runtime", "third_party"}
    for path in (root / "core").rglob("*.py"):
        if any(part in excluded for part in path.relative_to(root / "core").parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(errors="replace"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(name == "mcp_servers" or name.startswith(("mcp_servers.", "agent.")) for name in names), path


@pytest.mark.parametrize("service,entry", [
    ("worldengine_failure_analysis", "run.py"),
    ("sentry_video_search", "cli.py"),
])
def test_native_cli_runs_from_extracted_core_without_adagent(tmp_path, service, entry):
    core = tmp_path / "core"
    shutil.copytree(ROOT / "mcp_servers" / service / "core", core,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    result = subprocess.run([sys.executable, "-I", str(core / entry), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_agent_tools_use_only_canonical_adapter_service():
    expected = {
        "tool01_failure_analyzer": "worldengine_failure_analysis",
        "tool04_video_retriever": "sentry_video_search",
        "tool05_instance_discovery": "dggt",
        "tool06_reconstruction": "dggt",
        "tool08_edit_config_generator": "smart_edit",
        "tool09_scene_editor": "dggt",
        "tool13_policy_runner": "worldengine",
        "tool14_failure_case_filter": "worldengine",
        "tool15_policy_trainer": "worldengine",
    }
    for tool, service in expected.items():
        tree = ast.parse((ROOT / "tools" / tool / "real.py").read_text())
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("mcp_servers.")]
        assert modules == [f"mcp_servers.{service}.adapter.service"]
