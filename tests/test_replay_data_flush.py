"""Final native camera metadata must be persisted without inventing frames."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'mcp_servers/worldengine/adapter/replay_native.py'


def flush_function():
    function=next(node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node,ast.FunctionDef) and node.name=='flush_replay_data')
    namespace={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace[function.name]


def test_final_episode_is_saved_with_native_visualizer_disabled():
    frames={str(i):dict(timestamp=i) for i in range(24)}
    save=Mock()
    manager=SimpleNamespace(episode_data=frames,save_data=save)
    engine=SimpleNamespace(managers=dict(data_manager=manager))
    flush_function()(engine,dict(log_length=24))
    save.assert_called_once_with()
    assert manager.episode_data is frames and len(frames)==24


@pytest.mark.parametrize('count',[0,23,25])
def test_missing_or_extra_frames_prevent_completion(count):
    manager=SimpleNamespace(episode_data=dict.fromkeys(range(count)),save_data=Mock())
    with pytest.raises(RuntimeError,match='full requested time window'):
        flush_function()(SimpleNamespace(managers=dict(data_manager=manager)),dict(log_length=24))
    manager.save_data.assert_not_called()


@pytest.mark.parametrize('engine',[None,SimpleNamespace(managers={})])
def test_missing_data_manager_is_an_execution_error(engine):
    with pytest.raises(RuntimeError,match='no data manager'):
        flush_function()(engine,dict(log_length=24))
