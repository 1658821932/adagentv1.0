"""Reuse finished stages only when task, scene, map and policy still match."""
from pathlib import Path
import json
import sys

import pytest
from mcp_servers.scene_editing.core.adversarial import defense_probe_feedback,ego_stage_reuse


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def stage(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'defense_probe_feedback',defense_probe_feedback)
    scene=tmp_path/'scene';write(scene/'metadata/frames.json',[dict(timestamp=1)])
    args=dict(attack_result=dict(frozen_scene_path=str(scene),target_track='target'),
        target_effect='Stop safely',allow_safe_stop=True,use_foci=False)
    policy=dict(profile='test');source=tmp_path/'old'
    write(source/'input/stage_request.json',dict(stage='ego_editing',arguments=args))
    write(source/'output/defense_constructor/probe-binding.json',
        defense_probe_feedback.constructor_binding(scene,'target',policy,24))
    brief=source/'output/defense_understanding'
    write(brief/'config.json',dict(scene_dir=str(scene),target_track='target'))
    write(brief/'summary.json',dict(status='completed'))
    write(brief/'conflict-brief.json',dict(directions=[]))
    write(brief/'requests/old/request.json',dict(archived=True))
    return source,args,policy,scene


def test_artifacts_reused_without_old_api_requests(stage,tmp_path):
    source,args,policy,scene=stage;output=tmp_path/'new_brief'
    ego_stage_reuse.reuse_understanding(source,args,policy,24,output)
    assert (output/'conflict-brief.json').exists() and not (output/'requests').exists()
    assert (source/'output/defense_understanding/requests/old/request.json').exists()


def test_changed_task_rejected_before_copy(stage,tmp_path):
    source,args,policy,scene=stage
    args=dict(args,target_effect='Different task')
    with pytest.raises(ValueError,match='changed task/attack'):
        ego_stage_reuse.reuse_understanding(source,args,policy,24,tmp_path/'new')


def test_changed_scene_rejected_before_copy(stage,tmp_path):
    source,args,policy,scene=stage;write(scene/'metadata/frames.json',[dict(timestamp=2)])
    with pytest.raises(ValueError,match='scene/map/policy'):
        ego_stage_reuse.reuse_understanding(source,args,policy,24,tmp_path/'new')
