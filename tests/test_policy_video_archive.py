import os
import ast
from pathlib import Path
import time
from types import SimpleNamespace

from mcp_servers.worldengine.adapter.policy_videos import collect_policy_videos


def test_legacy_video_snapshots_survive_later_runs_reusing_the_filename(tmp_path):
    world=tmp_path/'world';source=world/'projects/SimEngine/dbg/video_visualizer/job/camera_scene.mp4'
    source.parent.mkdir(parents=True);source.write_bytes(b'first actual run')
    first=collect_policy_videos(tmp_path/'run1',world,time.time())
    assert len(first)==1 and Path(first[0]).read_bytes()==b'first actual run'
    source.write_bytes(b'next actual run')
    second=collect_policy_videos(tmp_path/'run2',world,time.time())
    assert Path(first[0]).read_bytes()==b'first actual run'
    assert Path(second[0]).read_bytes()==b'next actual run'
    assert first!=second


def test_run_owned_videos_are_preferred_over_other_jobs(tmp_path):
    result=tmp_path/'run';owned=result/'videos/scene.mp4';owned.parent.mkdir(parents=True);owned.write_bytes(b'owned')
    stray=tmp_path/'world/projects/SimEngine/dbg/video_visualizer/other/scene.mp4';stray.parent.mkdir(parents=True);stray.write_bytes(b'unrelated')
    assert collect_policy_videos(result,tmp_path/'world',time.time())==[str(owned)]


def test_expired_legacy_video_is_not_published(tmp_path):
    source=tmp_path/'world/projects/SimEngine/dbg/video_visualizer/job/scene.mp4';source.parent.mkdir(parents=True);source.write_bytes(b'old')
    os.utime(source,(1,1))
    assert collect_policy_videos(tmp_path/'run',tmp_path/'world',time.time())==[]


def test_native_visualizer_writes_each_run_to_its_owned_directory(tmp_path,monkeypatch):
    source=Path(__file__).resolve().parents[1]/'mcp_servers/worldengine/core/projects/SimEngine/worldengine/utils/agent_utils.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Video_visualizer')
    namespace=dict(os=os,Path=Path,WORLDENGINE_ROOT=str(tmp_path/'world'),get_engine=lambda:SimpleNamespace(global_config={'job_name':'same-scene','data_output_dir':str(tmp_path/'sensor_data')}))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    for run in ['run-a','run-b']:
        folder=tmp_path/run/'videos';monkeypatch.setenv('ADAGENT_NATIVE_VIDEO_DIR',str(folder))
        visualizer=namespace['Video_visualizer']()
        assert visualizer.path==str(folder) and folder.is_dir()
    explicit=namespace['Video_visualizer'](path=str(tmp_path/'explicit'))
    assert explicit.path==str(tmp_path/'explicit')
