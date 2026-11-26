"""Offline shared-API switching, including subprocess consumers and credential safety."""
import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import api_config
from scripts.api_profile import main


def test_client_import_does_not_read_credentials():
    path=Path(__file__).resolve().parents[1]/'agent/llm_client.py'
    spec=importlib.util.spec_from_file_location('isolated_llm_client_import',path)
    module=importlib.util.module_from_spec(spec)
    with (patch('dotenv.load_dotenv',side_effect=AssertionError('No credential reads at import')),
          patch('api_config.load_credentials',side_effect=AssertionError('No credential reads at import'))):
        spec.loader.exec_module(module)


@pytest.fixture
def api_file(tmp_path, monkeypatch):
    from common import api_profiles
    config = json.loads(api_config.API_CONFIG_PATH.read_text())
    config["active_profile"] = "kimi"
    config["tool_profiles"] = {}
    path = tmp_path / "api.json"
    path.write_text(json.dumps(config))
    env = tmp_path / ".env"
    env.write_text('API_KEY_1 = "offline-fake-credential-one"\nAPI_KEY_2=offline-fake-credential-two\n')
    monkeypatch.setattr(api_config, "API_CONFIG_PATH", path)
    monkeypatch.setattr(api_profiles, "API_CONFIG", path)
    monkeypatch.setattr(api_config, "ENV_PATH", env)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-original-credential")
    for name in ("API_KEY_1", "API_KEY_2"):
        # dotenv writes os.environ directly: ensure monkeypatch restores these
        # variables even when they did not exist before the test.
        monkeypatch.setenv(name, "fixture-placeholder")
        monkeypatch.delenv(name)
    return path


def add(path, name="api1", variable="API_KEY_1", model="model-one"):
    return main(["--config", str(path), "add", name,
                 "--base-url", "https://provider.invalid/v1", "--model", model,
                 "--key-env", variable])


def test_switch_reaches_main_editing_and_native_consumers(api_file, monkeypatch, capsys):
    from agent.llm_client import LLMClient
    from mcp_servers.scene_editing.adapter.service import backend_api_env
    from mcp_servers.sentry_video_search.core import api_settings as sentry
    from mcp_servers.worldengine_failure_analysis.core import api_settings as wfa

    assert add(api_file) == 0
    assert add(api_file, "api2", "API_KEY_2", "model-two") == 0
    original_retrieval = json.loads(api_file.read_text())["retrieval"]
    for name, variable, model, key in [
        ("api1", "API_KEY_1", "model-one", "offline-fake-credential-one"),
        ("api2", "API_KEY_2", "model-two", "offline-fake-credential-two"),
    ]:
        assert main(["--config", str(api_file), "use", name]) == 0
        with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=Mock(), Timeout=Mock())}):
            client = LLMClient()
        assert (client.model, client.base_url, client.api_key) == (model, "https://provider.invalid/v1", key)
        backend = Path(__file__).resolve().parents[1] / "mcp_servers/scene_editing/core/adversarial/editing_backend.py"
        injected = backend_api_env(["python", str(backend)], {})
        assert injected["ADVERSARIAL_API_MODEL"] == model
        assert injected["ADVERSARIAL_API_KEY"] == key
        # WFA reads the same JSON in a separate interpreter; legacy variables
        # must not silently change its model, URL or credential after selection.
        env = {variable: key, "OPENAI_MODEL": "stale-model", "OPENAI_BASE_URL": "https://stale.invalid/v1", "LLM_API_KEY": "wrong-key"}
        for module in (sentry, wfa):
            resolved = module.get_profile(config_path=api_file, environ=env, require_key=True)
            assert (resolved["model"], resolved["base_url"], resolved["api_key"]) == (model, "https://provider.invalid/v1", key)
        assert json.loads(api_file.read_text())["retrieval"] == original_retrieval
    assert main(["--config", str(api_file), "list"]) == 0
    assert main(["--config", str(api_file), "show"]) == 0
    printed = capsys.readouterr()
    persisted = api_file.read_text()
    for key in ("offline-fake-credential-one", "offline-fake-credential-two", "offline-original-credential"):
        assert key not in printed.out + printed.err + persisted


def test_missing_credential_does_not_switch(api_file, capsys):
    assert add(api_file, variable="MISSING_AUDIT_KEY") == 0
    before = api_file.read_bytes()
    assert main(["--config", str(api_file), "use", "api1"]) == 1
    assert api_file.read_bytes() == before
    assert "MISSING_AUDIT_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["retrieval", "active_profile", "reasoning_effort", "tool_profiles", "tool_efforts", "unknown"])
def test_rejects_non_reasoning_selection_without_mutation(api_file, name):
    before = api_file.read_bytes()
    assert main(["--config", str(api_file), "use", name]) == 1
    assert api_file.read_bytes() == before


def test_list_ignores_per_tool_and_global_settings(api_file, capsys):
    config = json.loads(api_file.read_text())
    config.update(reasoning_effort="medium", tool_profiles={"attack_editor": "kimi"},
                  tool_efforts={"attack_editor": "low"})
    api_file.write_text(json.dumps(config))
    before = api_file.read_bytes()
    assert main(["--config", str(api_file), "list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in rows} == {
        name for name, profile in config.items()
        if isinstance(profile, dict) and "model" in profile
    }
    assert api_file.read_bytes() == before


def test_scene_generator_reads_its_own_model_profile_and_effort(api_file, monkeypatch):
    from common import api_profiles

    config = json.loads(api_file.read_text())
    config['tool_profiles'] = {'scene_instance_generator': 'api1'}
    config['tool_efforts'] = {'scene_instance_generator': 'low'}
    api_file.write_text(json.dumps(config))
    monkeypatch.setattr(api_profiles, 'API_CONFIG', api_file)
    monkeypatch.delenv('SCENE_INSTANCE_GENERATOR_LLM_PROFILE', raising=False)
    monkeypatch.delenv('SCENE_INSTANCE_GENERATOR_REASONING_EFFORT', raising=False)
    assert api_profiles.profile_argument('scene_instance_generator') == 'api1'
    assert api_profiles.effort_for('scene_instance_generator') == 'low'


@pytest.mark.parametrize("name", ["reasoning_effort", "tool_profiles", "tool_efforts"])
def test_add_rejects_metadata_names_even_when_absent(api_file, capsys, name):
    config = json.loads(api_file.read_text())
    config.pop(name, None)
    api_file.write_text(json.dumps(config))
    before = api_file.read_bytes()
    assert add(api_file, name=name) == 1
    assert api_file.read_bytes() == before
    assert "reserved" in capsys.readouterr().err


def test_legacy_config_without_selector_still_uses_kimi(api_file):
    config = json.loads(api_file.read_text())
    config.pop("active_profile")
    api_file.write_text(json.dumps(config))
    assert api_config.get_profile(environ={})["model"] == config["kimi"]["model"]
    assert "api_key" not in api_config.get_profile(environ={})


@pytest.mark.parametrize("selection", ["missing", "retrieval", "active_profile", None, []])
def test_invalid_active_profile_never_falls_back_to_another_key(api_file, selection):
    config = json.loads(api_file.read_text())
    config["active_profile"] = selection
    api_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        api_config.get_profile(require_key=True, environ={"OPENAI_API_KEY": "wrong-fallback-key"})


@pytest.mark.parametrize("field,value", [
    ("api_key_env", "offline-fake-credential-one"),
    ("base_url", "https://provider.invalid/v1?key=offline-fake-credential-one"),
    ("base_url", "https://user:offline-fake-credential-one@provider.invalid/v1"),
])
def test_add_rejects_embedded_credentials_without_printing_them(api_file, capsys, field, value):
    args = ["--config", str(api_file), "add", "api1", "--base-url", "https://provider.invalid/v1",
            "--model", "model-one", "--key-env", "API_KEY_1"]
    flag = "--base-url" if field == "base_url" else "--key-env"
    args[args.index(flag) + 1] = value
    before = api_file.read_bytes()
    assert main(args) == 1
    assert api_file.read_bytes() == before
    printed = capsys.readouterr()
    assert "offline-fake-credential-one" not in printed.out + printed.err


def test_active_profile_cannot_contain_literal_key(api_file):
    config = json.loads(api_file.read_text())
    config["kimi"]["api_key"] = "offline-bad-literal"
    api_file.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="credentials in .env"):
        api_config.get_profile(environ={})


def test_dotenv_fallback_accepts_spaced_assignments(api_file, monkeypatch):
    original_import = builtins.__import__
    def without_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("offline fallback test")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_dotenv)
    api_config.load_credentials()
    import os
    assert os.environ["API_KEY_1"] == "offline-fake-credential-one"
