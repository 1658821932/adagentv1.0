"""Isolated task roots for service integration tests."""
import pytest

@pytest.fixture(autouse=True)
def service_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAGENT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("SENTRY_RUNTIME_ROOT", str(tmp_path / "native-runtime"))
    monkeypatch.delenv("ADAGENT_ACTIVE_SERVICE", raising=False)
    monkeypatch.delenv("ADAGENT_ACTIVE_RUN", raising=False)
