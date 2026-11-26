"""The public tool, direct adapter and launcher share the 1.5-7.5s default."""
import argparse
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from mcp_servers.worldengine.adapter import client, config
from mcp_servers.worldengine.adapter.launch import build_commands
from tools.tool06_policy_runner.schemas import Input


def test_collision_only_launcher_skips_products_but_preserves_closed_loop(tmp_path):
    args=argparse.Namespace(config='/cfg',checkpoint='/weights',model_name='p',
        data_type='s',react_type='NR',asset_name='s',added_name='s')
    env=dict(WORLDENGINE_ROOT=str(tmp_path),WORLDENGINE_TEST_PATH=str(tmp_path/'out'))
    _,alg,*_,normal=build_commands(args,env)
    sim,fast_alg,*_,fast=build_commands(args,{**env,'ADAGENT_COLLISION_ONLY':'true',
        'ADAGENT_COLLISION_TARGET_TRACK':'target','VISUALIZE_VIDEO':'true','VISUALIZE_BEV':'true'})
    assert alg==fast_alg
    assert fast['evaluation_horizon']==normal['evaluation_horizon']
    for key in ('with_metric_manager','save_data','visualize_video','visualize_BEV','with_dense_reward_manager'):
        assert fast['simulator'][key]=='false'
    for key in ('ego_controller','ego_policy','ego_client','ego_navigation','use_planner_actions',
                'num_history','num_future','sampling_poses'):
        assert fast['simulator'][key]==normal['simulator'][key]
    assert 'with_metric_manager=false' in sim
    assert normal['simulator']['with_metric_manager']==normal['simulator']['save_data']=='true'
    with pytest.raises(ValueError,match='require a target'):
        build_commands(args,{**env,'ADAGENT_COLLISION_ONLY':'true'})


def test_collision_scope_requires_complete_actual_coverage_and_correct_scene():
    from mcp_servers.worldengine.adapter.collision_report import collision_evaluation_scope
    report=dict(status='completed',target_collision_detected=False,frame_count=16,
        policy_takeover_step=3,policy_takeover_time_s=1.5,scene_id='scene')
    scope=collision_evaluation_scope(report,{'scene'},4,16)
    assert (scope['start_time_s'],scope['end_time_s'])==(1.5,7.5)
    assert scope['official_pdm_score'] is False
    for changed in ({'frame_count':12},{'policy_takeover_step':7},
                    {'status':'incomplete_target_coverage'},{'target_collision_detected':None}):
        with pytest.raises(RuntimeError,match='coverage'):
            collision_evaluation_scope({**report,**changed},{'scene'},4,16)
    with pytest.raises(RuntimeError,match='coverage'):
        collision_evaluation_scope(report,{'another_scene'},4,16)


@pytest.mark.parametrize('collision_only',[False,True])
def test_service_forwards_probe_mode_and_keeps_official_evaluation_separate(tmp_path,monkeypatch,collision_only):
    from types import SimpleNamespace
    from mcp_servers.worldengine.adapter import service
    from mcp_servers.worldengine.adapter.collision_report import summarize_frames
    cfg=tmp_path/'cfg.py';cfg.touch()
    weights=tmp_path/'weights.pth';weights.touch()
    script=tmp_path/'testing.sh';script.touch()
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT',str(tmp_path/'runtime'))
    monkeypatch.setenv('ADAGENT_COLLISION_ONLY','true')  # no inherited mode leakage into 07
    runtime=SimpleNamespace(project_root=tmp_path,timeout=60,testing_script=lambda:script)
    environments=[]
    def run_logged(command,**kwargs):
        env=kwargs['env']
        if 'WORLDENGINE_TEST_PATH' in env:
            environments.append(env)
            out=Path(env['WORLDENGINE_TEST_PATH']);(out/'WE_output').mkdir(parents=True)
            (out/'runner_report_test.json').write_text(json.dumps([dict(succeeded=True,scenario_name='scene')]))
            if collision_only:
                frames=[dict(scene_id='scene',step=i,time_s=i*.5,actors={
                    'ego':dict(box=[i,0,4,2,0],velocity=[2,0]),
                    'target':dict(box=[30,0,4,2,0],velocity=[0,0])}) for i in range(16)]
                (out/'WE_output/native_collision_report.json').write_text(json.dumps(summarize_frames(frames,'target',3)))
            else:
                (out/'WE_output/pdm_evaluation_scope.json').write_text(json.dumps(dict(
                    scope='full_rollout_pdm',scenes=[dict(scene_id='scene',status='completed')])))
        return CompletedProcess(command,0,'','')
    monkeypatch.setattr(service,'run_logged',run_logged)
    options=dict(num_history=4,max_frame_per_scene=16)
    if collision_only:options.update(collision_only=True,collision_target_track='target')
    result=service.run_simulation(str(cfg),str(weights),'p','s',options=options,config=runtime)
    scope=result['evaluation_scope']
    assert scope['scope']==('native_target_collision' if collision_only else 'full_rollout_pdm')
    assert environments[0].get('ADAGENT_COLLISION_ONLY')==('true' if collision_only else None)


def test_collision_probe_adapter_never_collects_video_or_training_data(tmp_path,monkeypatch):
    weights=tmp_path/'base.pth';weights.touch()
    cfg=tmp_path/'cfg.py';cfg.touch()
    monkeypatch.setattr(config,'DEFAULT_POLICY_CONFIG',cfg)
    monkeypatch.setattr(config,'DEFAULT_POLICY_CHECKPOINT',weights)
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT',str(tmp_path/'runtime'))
    sim=tmp_path/'simulation';(sim/'WE_output').mkdir(parents=True)
    (sim/'WE_output/native_collision_report.json').write_text(json.dumps(dict(
        status='completed',target_track='target',frame_count=16,target_collision_detected=False)))
    requests=[]
    def run_logged(command,**kwargs):
        requests.append(json.loads(kwargs['input']))
        return CompletedProcess(command,0,json.dumps(dict(status='success',result=dict(
            result_dir=str(sim),reports=[{'succeeded':True}]))),'')
    def unexpected(*args,**kwargs):raise AssertionError('Unexpected scoring/video collection')
    monkeypatch.setattr(client,'run_logged',run_logged)
    monkeypatch.setattr(client,'read_pdm_subscores',unexpected)
    monkeypatch.setattr('mcp_servers.worldengine.adapter.policy_videos.collect_policy_videos',unexpected)
    monkeypatch.setattr(client,'summarize_policy_behavior',lambda *a,**kw:{'stopped_detected':True})
    result=client.run_policy(dict(collision_only=True,collision_target_track='target',visualize_video=True))
    assert len(requests)==1
    assert requests[0]['collision_only'] is True and requests[0]['visualize_video'] is False
    assert result['policy_behavior_summary']['stopped_detected'] is True
    assert 'pdm_subscores' not in result and 'videos' not in result and 'rlft_training_data' not in result


@pytest.mark.parametrize('dense', [False, True])
def test_schema_and_launcher_share_standard_window(tmp_path, dense):
    public = Input(with_dense_reward_manager=dense)
    assert (public.num_history, public.max_frame_per_scene) == (4, 16)
    args = argparse.Namespace(config='/cfg', checkpoint='/weights', model_name='p',
        data_type='s', react_type='NR', asset_name='s', added_name='s')
    env = dict(WORLDENGINE_ROOT=str(tmp_path), WORLDENGINE_TEST_PATH=str(tmp_path/'out'),
        WITH_DENSE_REWARD_MANAGER=str(dense).lower())
    sim, alg, *_, effective = build_commands(args, env)
    horizon = effective['evaluation_horizon']
    assert (effective['num_history'], effective['num_future'], effective['max_frame_per_scene']) == (4, 12, 16)
    assert (horizon['start_step']*.5, horizon['end_step']*.5) == (1.5, 7.5)
    assert 'sampling_poses=12' in sim and 'buffer_size=13' in sim
    assert 'sim.max_frame_per_scene=16' in alg
    assert not any('reward_sampling_poses=' in value for value in sim)


@pytest.mark.parametrize('dense', [False, True])
@pytest.mark.parametrize('explicit_frames', [None, 24])
def test_direct_adapter_does_not_derive_default_from_long_source(tmp_path, monkeypatch, dense, explicit_frames):
    scene = tmp_path/'expert'
    (scene/'metadata').mkdir(parents=True)
    (scene/'metadata/frames.json').write_text(json.dumps({'frame_count':40}))
    weights = tmp_path/'base.pth'; weights.write_bytes(b'offline fixture')
    cfg = tmp_path/'policy.py'; cfg.write_text('# offline fixture')
    monkeypatch.setattr(config, 'DEFAULT_POLICY_CONFIG', cfg)
    monkeypatch.setattr(config, 'DEFAULT_POLICY_CHECKPOINT', weights)
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT', str(tmp_path/'runtime'))
    simulation = tmp_path/'simulation'
    rollout = simulation/'WE_output/openscene_format'
    rollout.mkdir(parents=True)
    if dense:
        rewards = rollout/'pdms_pkl'; rewards.mkdir()
        (rewards/'scene_step_3_scores.pkl').write_bytes(b'presence fixture')
        plans = simulation/'plan_traj'; plans.mkdir()
        (plans/'plan_idx.csv').write_text('prefix,step,plan_idx\nscene,4,7\n')
    requests = []

    def run_logged(command, **kwargs):
        if 'input' in kwargs:
            requests.append(json.loads(kwargs['input']))
            result = {'result_dir':str(simulation), 'reports':[{'succeeded':True}]}
            return CompletedProcess(command, 0, json.dumps({'status':'success','result':result}), '')
        return CompletedProcess(command, 0, json.dumps({'rlft_training_data':None,
            'rlft_data_report':{'status':'no_eligible_scenes'}}), '')

    monkeypatch.setattr(client, 'run_logged', run_logged)
    args = dict(scenario_data_folder_path=str(scene), with_dense_reward_manager=dense)
    if explicit_frames is not None:
        args['max_frame_per_scene'] = explicit_frames
    source = dict(args)
    result = client.run_policy(args)
    assert result['policy_result'] == 'completed'
    assert requests[0]['num_history'] == 4
    assert requests[0]['max_frame_per_scene'] == (explicit_frames or 16)
    assert args == source
