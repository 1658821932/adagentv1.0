"""Protect the core/adapter boundary and standalone command line entry points."""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ["scene_editing", "sentry_video_search", "worldengine", "worldengine_failure_analysis"]


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
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            assert not any(name in {"mcp_servers", "agent", "api_config", "tools", "common"} or name.startswith(("mcp_servers.", "agent.", "tools.", "common.")) for name in names), path


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
        "tool03_video_retriever": "sentry_video_search",
        "tool05_attack_vehicle_editor": "scene_editing",
        "tool06_ego_trajectory_editor": "scene_editing",
        "tool07_scene_replayer": "worldengine",
        "tool08_policy_runner": "worldengine",
        "tool09_failure_case_filter": "worldengine",
        "tool10_policy_trainer": "worldengine",
    }
    for tool, service in expected.items():
        tree = ast.parse((ROOT / "tools" / tool / "real.py").read_text())
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("mcp_servers.")]
        assert modules == [f"mcp_servers.{service}.adapter.service"]



def test_extracted_sentry_resolves_models_and_retrieval_credentials(tmp_path, monkeypatch):
    import json
    core=tmp_path/"core"
    shutil.copytree(ROOT/"mcp_servers/sentry_video_search/core",core,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    profiles=json.loads((core/"api.example.json").read_text())
    profiles["kimi"].update(model="extracted-test",base_url="https://standalone.invalid/v1",api_key_env="NATIVE_KIMI_KEY")
    profiles["retrieval"]["api_key_env"]="NATIVE_RETRIEVAL_KEY"
    config=tmp_path/"api.json";config.write_text(json.dumps(profiles))
    monkeypatch.setenv("SENTRY_API_CONFIG",str(config))
    monkeypatch.setenv("NATIVE_KIMI_KEY","offline-kimi-key")
    monkeypatch.setenv("NATIVE_RETRIEVAL_KEY","offline-retrieval-key")
    for key in ("OPENAI_BASE_URL","LLM_BASE_URL","OPENAI_MODEL","LLM_MODEL"):
        monkeypatch.delenv(key,raising=False)
    script = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "from core.service import _ensure_dashscope_key;"
        "from core.pipeline.common import model_settings;"
        "_ensure_dashscope_key();"
        "assert model_settings()==('extracted-test','https://standalone.invalid/v1','offline-kimi-key');"
        "assert 'api_config' not in sys.modules"
    )
    result=subprocess.run([sys.executable,"-I","-c",script,str(tmp_path)],
                          cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0,result.stderr
