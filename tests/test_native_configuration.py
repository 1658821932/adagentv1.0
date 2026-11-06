"""Exercise extracted native configuration without the host package on sys.path."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("service,variable", [
    ("sentry_video_search", "SENTRY_API_CONFIG"),
    ("worldengine_failure_analysis", "WFA_API_CONFIG"),
])
def test_extracted_native_profile_resolution_and_scoped_host_injection(tmp_path, service, variable):
    core = tmp_path / "native"
    core.mkdir()
    for name in ("api_settings.py", "api.example.json"):
        shutil.copy2(ROOT / "mcp_servers" / service / "core" / name, core / name)
    script = core / "check.py"
    script.write_text("""
import os,json
from pathlib import Path
from api_settings import get_profile,bind_profiles,vlm_settings
p=Path(__file__).parent/"local.json"
p.write_text(json.dumps({"kimi":{"base_url":"https://native.invalid/v1","model":"native-model","api_key_env":"NATIVE_TEST_KEY"}}))
os.environ[VARIABLE]=str(p)
os.environ["NATIVE_TEST_KEY"]="offline-test-key"
assert vlm_settings()=={"endpoint":"https://native.invalid/v1/chat/completions","model":"native-model","api_key":"offline-test-key"}
with bind_profiles(lambda *a,**k: {"model":"injected"}):
 assert get_profile()["model"]=="injected"
assert get_profile()["model"]=="native-model"
assert "api_config" not in __import__("sys").modules
""".replace("VARIABLE", repr(variable)))
    # -I excludes PYTHONPATH and cwd; add only the extracted native directory.
    command = "import runpy,sys;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2])"
    proc = subprocess.run([sys.executable, "-I", "-c", command, str(core), str(script)],
                          cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert proc.returncode == 0, proc.stderr


def test_shared_contract_layer_never_imports_tools_or_adapters():
    import ast
    for path in (ROOT / "common").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(name.split(".")[0] in {"tools","agent","mcp_servers","api_config"} for name in names), path
