"""Native policy evaluation must use the same edited-scene coordinates as replay."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
import importlib.util

from mcp_servers.worldengine.adapter.native_simulation import activate_exported_scene_adapter


def test_edited_scene_activates_pose_calibration_once(monkeypatch):
    install = Mock()
    monkeypatch.setitem(sys.modules,'replay_pose',SimpleNamespace(install=install))
    scenes = {name:dict(metadata=dict(native_export=dict(version='edited_scene.v2',ego_calibration_frame='reconstruction')))
              for name in ('first','second')}
    assert activate_exported_scene_adapter(scenes) == ['first','second']
    install.assert_called_once_with()


def test_legacy_utm_export_is_rejected_before_loading_renderer(monkeypatch):
    import pytest
    install=Mock()
    monkeypatch.setitem(sys.modules,'replay_pose',SimpleNamespace(install=install))
    with pytest.raises(ValueError,match='must be re-exported'):
        activate_exported_scene_adapter({'legacy':{'metadata':{'native_export':{'version':'edited_scene.v1','ego_calibration_frame':'global'}}}})
    install.assert_not_called()


def test_original_dataset_does_not_activate_edited_pose_adapter(monkeypatch):
    install = Mock(side_effect=AssertionError('Original dataset must retain its renderer configuration'))
    monkeypatch.setitem(sys.modules,'replay_pose',SimpleNamespace(install=install))
    assert activate_exported_scene_adapter(dict(original=dict(metadata={}))) == []
    install.assert_not_called()


def test_planner_monitor_waits_for_actual_scene_end(tmp_path,monkeypatch):
    from mcp_servers.worldengine.adapter.native_planner import configure_monitor
    root=Path(__file__).resolve().parents[1]
    source=root/'mcp_servers/worldengine/core/projects/AlgEngine/closed_loop/monitor.py'
    spec=importlib.util.spec_from_file_location('closed_loop.monitor',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules,'closed_loop',SimpleNamespace(monitor=module))
    monkeypatch.setitem(sys.modules,'closed_loop.monitor',module)
    for i in range(4): (tmp_path/f'case3_{i}.pkl').touch()
    configure_monitor(24)
    logger=Mock()
    monitor=module.AsyncFileMonitor(str(tmp_path),max_history=4,max_frame_per_scene=12,
        wait_for_files=True,logger=logger)
    monitor.file_history.append(str(tmp_path/'case3_11.pkl'))
    assert monitor._reinitialize_queue() is False
    assert monitor.scene_idx == 0 and monitor.prefix == 'case3'
    monitor.file_history.append(str(tmp_path/'case3_23.pkl'))
    monkeypatch.setattr(monitor,'_initialize_queue',lambda:True)
    assert monitor._reinitialize_queue() is True
    assert monitor.scene_idx == 1
