"""PDM-safe attacks reenter tool05 with compact feedback and unchanged task gates."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from mcp_servers.scene_editing.adapter.schemas import AttackInput
from scripts.run_adversarial_validation import load_seed, pdm_feedback, run


def config():
    source=dict(scene_id='test',scene_token='scene-token',sample_root='/original/scene',dataset='nuplan_openscene',
        matched_camera='CAM_F0',frame_count=24,fps=2.,manifest_path='/original/map.json',
        frames_path='/original/frames.json',cameras_path='/original/cameras.json',cameras={},images_complete=True)
    return dict(attack_input=dict(scene_ref=source,instruction='Add a stationary vehicle ahead',
        operation='add_vehicle',maneuver='stationary',vehicle=dict(asset_path='/asset.ply',size_m=[4.,2.,1.5]),
        task_contract=dict(objective=dict(interaction='allowed'))),target_effect='Stop safely',use_foci=False,
        budget=dict(max_attack_attempts=2,max_api_requests=40,total_seconds=7200))


def envelope(tool,result):
    return dict(status='success',tool=tool,result=result)


def attack_result(index=1):
    source=AttackInput.model_validate(config()['attack_input']).model_dump()['scene_ref']
    return envelope('attack_vehicle_editor',dict(selected_candidate_id='seed_000',target_track='target',
        frozen_scene_path=f'/output/attack_{index}',submission_path='/missing/trajectory_editing/output.json',source_scene_ref=source))


def pdm_result(safe=True):
    return envelope('policy_runner',dict(pdm_challenge=dict(pdm_defense_succeeded=safe,
        baseline=dict(contact_samples=0,road_violation_samples=0),
        attack=dict(contact_samples=0 if safe else 1,road_violation_samples=0,min_target_gap_m=.425 if safe else 0.))))


def test_safe_pdm_revises_then_reaches_defense_and_exact_video(tmp_path):
    calls=[]
    attacks=0
    pdms=0
    def tools(tool,arguments,mode):
        nonlocal attacks,pdms
        assert mode=='real'
        calls.append((tool,deepcopy(arguments)))
        if tool=='attack_vehicle_editor':
            attacks+=1
            return attack_result(attacks)
        if tool=='policy_runner':
            pdms+=1
            return pdm_result(pdms==1)
        if tool=='ego_trajectory_editor':
            assert arguments['attack_result']['result']['frozen_scene_path']=='/output/attack_2'
            return envelope(tool,dict(rendered_videos=dict(manual_explained='/output/attack_2/manual.mp4')))
    value=config()
    result=run(value,tmp_path/'run',tool_runner=tools)
    assert result['status']=='accepted'
    assert [name for name,_ in calls]==['attack_vehicle_editor','policy_runner','attack_vehicle_editor',
        'policy_runner','ego_trajectory_editor']
    first,second=calls[0][1],calls[2][1]
    assert second['revision_feedback']['outcome']=='pdm_defense_succeeded'
    assert second['revision_feedback']['evidence']['pdm']['attack']['min_target_gap_m']==.425
    assert {k:v for k,v in second.items() if k!='revision_feedback'}=={k:v for k,v in first.items() if k!='revision_feedback'}
    assert value==config()


def test_safe_pdm_stops_at_attempt_budget_without_defense(tmp_path):
    calls=[]
    def tools(tool,arguments,mode):
        calls.append(tool)
        return attack_result() if tool=='attack_vehicle_editor' else pdm_result()
    value=config();value['budget']['max_attack_attempts']=1
    result=run(value,tmp_path/'run',tool_runner=tools)
    assert result['status']=='no_accepted_candidate'
    assert result['stop_reason']=='attack_attempt_budget_exhausted'
    assert calls==['attack_vehicle_editor','policy_runner']


def test_pdm_execution_error_never_triggers_stronger_attack(tmp_path):
    calls=[]
    def tools(tool,arguments,mode):
        calls.append(tool)
        return attack_result() if tool=='attack_vehicle_editor' else dict(status='error',tool=tool,error=dict(type='connection_error'))
    result=run(config(),tmp_path/'run',tool_runner=tools)
    assert result['status']=='paused'
    assert calls==['attack_vehicle_editor','policy_runner']


def test_seed_counts_as_attempt_and_already_used_requests(tmp_path):
    seed=dict(directory='/prior/run',attack=attack_result(),pdm=pdm_result(),api_requests=4)
    seen=[]
    def tools(tool,arguments,mode):
        seen.append((tool,arguments))
        return attack_result(2) if tool=='attack_vehicle_editor' else pdm_result()
    result=run(config(),tmp_path/'run',seed=seed,tool_runner=tools)
    assert result['status']=='no_accepted_candidate'
    assert len(result['attempts'])==2 and result['attempts'][0]['reused']
    assert result['api_requests']==4
    assert seen[0][1]['revision_feedback']['outcome']=='pdm_defense_succeeded'
    assert [n for n,_ in seen]==['attack_vehicle_editor','policy_runner']


def test_inconsistent_safe_pdm_is_not_forwarded():
    pdm=pdm_result();pdm['result']['pdm_challenge']['attack']['contact_samples']=1
    with pytest.raises(ValueError,match='contradicts'):
        pdm_feedback(attack_result(),pdm)


def test_seed_binding_rejects_another_attack_version(tmp_path):
    value=config()['attack_input'];attack=attack_result()
    files={'attack-input.json':value,'attack-result.json':attack,'pdm-result.json':pdm_result(),
        'pdm-input.json':dict(challenge_baseline_scene_path='/original/scene',
            challenge_attack_scene_path='/different/frozen',challenge_target_track='target',
            challenge_target_is_new=True,challenge_history_s=3.)}
    for name,data in files.items():(tmp_path/name).write_text(json.dumps(data))
    with pytest.raises(ValueError,match='different attack or timeline'):
        load_seed(tmp_path,value)


def test_loop_forwards_explicit_traffic_light_mode(tmp_path):
    value=config();value['attack_input']['pdm_ignore_traffic_lights']=True
    seen=[]
    def tools(tool,arguments,mode):
        seen.append((tool,arguments))
        return attack_result() if tool=='attack_vehicle_editor' else pdm_result()
    run(value,tmp_path/'run',tool_runner=tools)
    assert all(args['challenge_ignore_traffic_lights'] for name,args in seen if name=='policy_runner')


def test_unsafe_original_diagnostic_does_not_block_edited_scene_feedback():
    pdm=pdm_result()
    pdm['result']['pdm_challenge']['baseline']['contact_samples']=12
    assert pdm_feedback(attack_result(),pdm)['outcome']=='pdm_defense_succeeded'


def test_attack_business_refusal_ends_task_without_pdm_or_defense(tmp_path):
    calls=[]
    def tools(tool,arguments,mode):
        calls.append(tool)
        return envelope(tool,dict(status='unable_to_complete',reason='No aligned merge found',
                                   scope='current_tools_and_budget'))
    result=run(config(),tmp_path/'run',tool_runner=tools)
    assert result['status']=='unable_to_complete'
    assert result['attempts'][0]['status']=='attack_refused'
    assert calls==['attack_vehicle_editor']


def test_defense_business_failure_revises_attack_under_original_contract(tmp_path):
    calls=[]
    count=0
    def tools(tool,arguments,mode):
        nonlocal count
        calls.append((tool,deepcopy(arguments)))
        if tool=='attack_vehicle_editor':
            count+=1
            if count==2:
                assert arguments['revision_feedback']['stage']=='defense'
                assert arguments['task_contract']==config()['attack_input']['task_contract']
                return envelope(tool,dict(status='unable_to_complete'))
            return attack_result()
        if tool=='policy_runner':return pdm_result(False)
        assert tool=='ego_trajectory_editor'
        return envelope(tool,dict(status='defense_not_found',
            detail=dict(status='not_submitted',stop_reason='max_turns',model_actions=10)))
    result=run(config(),tmp_path/'run',tool_runner=tools)
    assert result['status']=='unable_to_complete'
    assert [name for name,_ in calls]==['attack_vehicle_editor','policy_runner','ego_trajectory_editor','attack_vehicle_editor']


def test_render_execution_error_preserves_attack_without_reattack(tmp_path):
    calls=[]
    def tools(tool,arguments,mode):
        calls.append(tool)
        if tool=='attack_vehicle_editor':return attack_result()
        if tool=='policy_runner':return pdm_result(False)
        return dict(status='error',tool=tool,error=dict(type='render_error'))
    result=run(config(),tmp_path/'run',tool_runner=tools)
    assert result['status']=='paused'
    assert calls==['attack_vehicle_editor','policy_runner','ego_trajectory_editor']


def test_defense_revision_keeps_business_failure_reason_and_evidence():
    from scripts.run_adversarial_validation import defense_revision_feedback
    response=dict(status='defense_not_found',stop_reason='model_turn_budget_exhausted',
        reason='Every attempted stop exceeds road boundary',failed_requirements=['official_road'],
        evidence_paths=['/output/defense/summary.json'],detail=dict(status='not_submitted'))
    feedback=defense_revision_feedback(response,attack_result())
    assert feedback['evidence']['reason']==response['reason']
    assert feedback['evidence']['failed_requirements']==['official_road']
    assert feedback['evidence']['evidence_paths']==response['evidence_paths']


def test_missing_foci_setting_defaults_to_enabled_in_registered_chain(tmp_path):
    value=config();value.pop('use_foci')
    observed=[]
    def tools(tool,arguments,mode):
        if tool=='attack_vehicle_editor':return attack_result()
        if tool=='policy_runner':return pdm_result(False)
        if tool=='ego_trajectory_editor':
            observed.append(arguments['use_foci'])
            return envelope(tool,dict(rendered_videos=dict(manual_explained='/new/video.mp4')))
        return envelope(tool,dict(passed=True))
    assert run(value,tmp_path/'run',tool_runner=tools)['status']=='accepted'
    assert observed==[True]
