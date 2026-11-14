"""The public tool, direct adapter and launcher share the 1.5-7.5s default."""
import argparse
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from mcp_servers.worldengine.adapter import client, config
from mcp_servers.worldengine.adapter.launch import build_commands
from tools.tool07_policy_runner.schemas import Input


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
