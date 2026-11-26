"""Whole-scene evaluation must include late collisions without changing planning."""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mcp_servers.worldengine.adapter.evaluation_horizon import full_horizon,validate_metric_inputs,install_metric_audit


@pytest.mark.parametrize('history,frames',[(4,12),(4,24),(4,40),(6,30)])
def test_single_derived_window_covers_every_state_after_takeover(history,frames):
    d=full_horizon(history,frames)
    assert d['sampling_poses']==frames-history
    assert d['buffer_size']==frames-history+1
    assert d['start_step']==history-1 and d['end_step']==frames-1


def test_old_eight_interval_override_cannot_silently_shorten_case3():
    with pytest.raises(ValueError,match='stale override'):full_horizon(4,24,8)


def manager():
    return SimpleNamespace(get_pred_states=lambda:np.zeros((21,11)),
        _cached_expert_traj=np.zeros((21,3)),_cached_ego_velocities=np.zeros((21,2)),
        detection_tracks_list=[{} for _ in range(21)],
        _proposal_sampling=SimpleNamespace(num_poses=20),buffer_size=21,current_step=23,
        current_scene=dict(id='case3',sample_rate=10))


@pytest.mark.parametrize('field',['_cached_expert_traj','_cached_ego_velocities','detection_tracks_list'])
def test_short_metric_sequence_is_rejected(field):
    m=manager();setattr(m,field,[None]*9)
    with pytest.raises(RuntimeError,match='sequence lengths differ'):validate_metric_inputs(m,full_horizon(4,24))


def test_actual_metric_pred_states_include_the_late_contact():
    root=Path(__file__).resolve().parents[1]
    source=root/'mcp_servers/worldengine/core/projects/SimEngine/worldengine/manager/metric_manager.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MetricManager')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_pred_states')
    namespace=dict(np=np)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
    states=[SimpleNamespace(rear_axle=SimpleNamespace(x=i*.5,y=0,heading=0)) for i in range(24)]
    m=SimpleNamespace(ego_states_list=states,num_history=4,buffer_size=21)
    actual=namespace['get_pred_states'](m)
    assert len(actual)==21 and actual[0,0]==1.5 and actual[-1,0]==11.5
    assert 7.5 in actual[:,0]


def test_audit_records_real_sequences_and_metric_completion(tmp_path,monkeypatch):
    class Metric:
        compute_pdm_scores=Mock(return_value='scores')
    monkeypatch.setitem(sys.modules,'worldengine',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'worldengine.manager',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'worldengine.manager.metric_manager',SimpleNamespace(MetricManager=Metric))
    install_metric_audit(tmp_path,full_horizon(4,24))
    assert Metric.compute_pdm_scores(manager(),'reference','velocity')=='scores'
    report=json.loads((tmp_path/'pdm_evaluation_scope.json').read_text())
    row=report['scenes'][0]
    assert row['start_time_s']==1.5 and row['end_time_s']==11.5
    assert set(row['sequence_counts'].values())=={21}


def test_shared_launcher_changes_scoring_not_vad_predictor_horizon(tmp_path):
    from mcp_servers.worldengine.adapter.launch import build_commands
    args=argparse.Namespace(config='/cfg',checkpoint='/weights',model_name='p',data_type='s',react_type='NR',asset_name='s',added_name='s')
    env=dict(WORLDENGINE_ROOT=str(tmp_path),WORLDENGINE_TEST_PATH=str(tmp_path/'out'),
        NUM_HISTORY='4',MAX_FRAME_PER_SCENE='24',NUM_FUTURE='20')
    sim,alg,*_,effective=build_commands(args,env)
    assert 'sampling_poses=20' in sim and 'buffer_size=21' in sim
    assert not any('planning_steps' in arg or 'planning_head' in arg for arg in alg)
    assert effective['evaluation_horizon']['scope']=='full_rollout_pdm'
    assert not any('reward_sampling_poses' in arg or 'reward_buffer_size' in arg for arg in sim)


def test_public_native_schema_derives_window_and_rejects_stale_manual_override():
    from tools.tool06_policy_runner.schemas import Input
    arguments=dict(config_path='/cfg',checkpoint_path='/weights',model_name='p',data_type='s',
        max_frame_per_scene=24,num_history=4,num_future=20)
    assert Input.model_validate(arguments).sampling_poses is None
    with pytest.raises(ValueError,match='stale override'):Input.model_validate(dict(arguments,sampling_poses=8))
