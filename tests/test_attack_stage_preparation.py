"""Separate deterministic acceleration and Agent actor editing, without API/GPU calls."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
sys.path.insert(0,str(CORE))
import acceptance_policy as policy
from attack_ego_preparation import prepare_acceleration,history_state
from attack_configuration import same_scene_constraints
from ego_history import verify


@pytest.fixture
def source(tmp_path):
    scene=tmp_path/'source';(scene/'metadata').mkdir(parents=True)
    frames=[]
    for i in range(24):
        pose=np.eye(4);pose[:2,3]=[100.+i*.5,200.]
        frames.append(dict(timestamp=1000000+i*500000,token=str(i),ego2global=pose.tolist(),
            ego2global_rotation=[1.,0.,0.,0.],lidar2ego=np.eye(4).tolist(),
            anns=dict(track_tokens=['other'],gt_names=['vehicle'],
                gt_boxes=[[30-i*.5,8,0,4,2,1.5,0]],gt_velocity_3d=[[0.,0.,0.]])))
    (scene/'metadata/frames.json').write_text(json.dumps(frames))
    road={'features':{'lane':{'polygon':[[-10,-20],[100,-20],[100,20],[-10,20]]}}}
    (scene/'metadata/map_drivable_official.json').write_text(json.dumps(road))
    return scene


def test_automatic_preparation_is_bounded_and_has_no_agent_calls(source,tmp_path):
    before=(source/'metadata/frames.json').read_bytes()
    report=prepare_acceleration(source,tmp_path/'prepare',policy.for_new_run())
    assert report['status']=='prepared' and report['api_calls']==report['policy_tests']==0
    assert len(report['attempts'])<=3 and report['shared_state']['frozen'] is True
    assert report['shared_state']['takeover']['speed_mps']>history_state(source)['takeover']['speed_mps']
    assert verify(report['scene'])
    assert (source/'metadata/frames.json').read_bytes()==before


def test_failed_preferences_record_each_attempt_and_keep_source(source,tmp_path,monkeypatch):
    import attack_ego_preparation as module
    calls=[]
    def impossible(*args,**kwargs):
        calls.append(args[2]);raise ValueError('history contacts a parked actor')
    monkeypatch.setattr(module,'prepare',impossible)
    report=prepare_acceleration(source,tmp_path/'prepare',policy.for_new_run())
    assert report['status']=='unchanged' and Path(report['scene'])==source
    assert len(calls)==3 and len(report['attempts'])==3
    speeds=[value[0]['speed_mps'] for value in calls]
    assert speeds[0]>speeds[1]>speeds[2]>1.
    assert all('parked actor' in row['reason'] for row in report['attempts'])


def test_explicit_unreachable_speed_is_not_silently_lowered(source,tmp_path):
    report=prepare_acceleration(source,tmp_path/'prepare',policy.for_new_run(),target_speed_mps=14.)
    assert report['status']=='unavailable' and len(report['attempts'])==1
    assert report['attempts'][0]['target_speed_mps']==14.
    assert 'acceleration' in report['attempts'][0]['reason']


def test_disabled_and_resumed_preparations_preserve_the_exact_history(source,tmp_path):
    kept=prepare_acceleration(source,tmp_path/'off',policy.for_new_run(),enabled=False)
    assert kept['status']=='unchanged' and kept['attempts']==[]
    first=prepare_acceleration(source,tmp_path/'first',policy.for_new_run())
    resumed=prepare_acceleration(source,tmp_path/'resume',policy.for_new_run(),previous_scene=first['scene'])
    branch=prepare_acceleration(first['scene'],tmp_path/'object_switch',policy.for_new_run())
    for value in (resumed,branch):
        assert value['status']=='reused' and value['attempts']==[]
        assert value['shared_state']==first['shared_state']
        assert (Path(value['scene'])/'metadata/frames.json').read_bytes()==(Path(first['scene'])/'metadata/frames.json').read_bytes()


def test_later_reverse_tail_does_not_block_the_contiguous_forward_history(source,tmp_path):
    path=source/'metadata/frames.json';frames=json.loads(path.read_text())
    for i in range(20,24):frames[i]['ego2global'][0][3]=109.5-.01*(i-19)
    path.write_text(json.dumps(frames))
    report=prepare_acceleration(source,tmp_path/'prepare',policy.for_new_run())
    assert report['status']=='prepared'
    manifest=json.loads((Path(report['scene'])/'ego-history.json').read_text())
    assert manifest['source_route_trimmed_before_reverse'] is True
    assert manifest['source_route_end_s']==9.5
    assert len(manifest['original_navigation_ego2global'])==24
    assert verify(report['scene'])


def test_history_adjustment_restarts_from_source_and_remains_explicit(source,tmp_path):
    first=prepare_acceleration(source,tmp_path/'first',policy.for_new_run())
    revised=prepare_acceleration(first['scene'],tmp_path/'revision',policy.for_new_run(),target_speed_mps=2.)
    assert revised['status']=='prepared'
    manifest=json.loads((Path(revised['scene'])/'ego-history.json').read_text())
    assert Path(manifest['source_scene'])==source and manifest['targets']==[{'time_s':1.5,'speed_mps':2.}]
    assert revised['shared_state']['initial_speed_mps']==first['shared_state']['initial_speed_mps']


def test_explicit_ego_speed_can_reduce_takeover_speed(source, tmp_path):
    prepared = prepare_acceleration(source, tmp_path/'slower', policy.for_new_run(), target_speed_mps=.65)
    assert prepared['status'] == 'prepared'
    assert prepared['shared_state']['takeover']['speed_mps'] < history_state(source)['takeover']['speed_mps']


def test_other_vehicle_prompts_receive_only_the_final_ego_state():
    from attack_context_view import editing_view, selection_view

    state = dict(time_s=1.5, rear_axle_xy_m=[2.3, -.2], yaw_rad=-.15, speed_mps=2.23)
    inputs = dict(constraints=dict(prepare_ego_history=True, ego_history_route={'end_xy_m': [2.35, -.18]},
        _fixed_ego_history=dict(frozen=True, end_s=1.5, status='prepared', takeover=state,
                                initial_speed_mps=.75, report_path='stage-a-private.json')))
    for view in (editing_view(inputs), selection_view(inputs)):
        assert 'ego_history_route' not in view['constraints']
        assert 'prepare_ego_history' not in view['constraints']
        assert view['fixed_ego_history'] == dict(frozen=True, end_s=1.5, status='prepared', takeover=state)


def test_stage_a_completes_before_any_actor_selection_and_does_not_use_api(source,tmp_path,monkeypatch):
    import run_attack_workflow as workflow
    monkeypatch.setenv('ADVERSARIAL_EXPERIMENTS_ROOT',str(tmp_path))
    monkeypatch.delenv('ADVERSARIAL_ATTACK_SESSION',raising=False)
    monkeypatch.setattr(workflow,'experiments_root',lambda:tmp_path)
    monkeypatch.setattr(workflow,'OpenAI',lambda **kw:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(workflow,'load_llm_profile',lambda:dict(api_key='offline',base_url='http://offline.invalid'))
    monkeypatch.setattr(workflow,'StreamingRequestArchive',lambda *args:SimpleNamespace())
    selected=[]
    def select(scene,folder,instruction,assets,constraints,*args):
        selected.append(scene)
        assert (scene/'ego-history.json').is_file()
        assert constraints['_fixed_ego_history']['frozen']
        folder.mkdir(parents=True)
        record=dict(status='not_submitted',stop_reason='finished_without_object_selection')
        (folder/'output.json').write_text(json.dumps(record))
        return record
    monkeypatch.setattr(workflow,'select_object',select)
    result=workflow.run(source,tmp_path/'attack','Construct a late-braking challenge.',constraints={'prepare_ego_history':True})
    assert result['ego_history_preparation']['status']=='prepared' and len(selected)==1
    bad=workflow.run(source,tmp_path/'bad','Construct a challenge.',constraints=dict(prepare_ego_history=True,ego_history_target_speed_mps=14.))
    assert len(selected)==1 and bad['stop_reason']=='ego_history_unavailable'
    from business_outcome import bounded_outcome
    refused=bounded_outcome('attack_editing',bad,tmp_path/'bad')
    assert refused['status']=='unable_to_complete' and 'Explicit' in refused['reason']


def test_workflow_resume_reuses_history_and_shared_probe_ledger(source,tmp_path,monkeypatch):
    import run_attack_workflow as workflow
    from attack_editor import write_trajectory
    monkeypatch.setenv('ADVERSARIAL_EXPERIMENTS_ROOT',str(tmp_path))
    monkeypatch.setenv('ADVERSARIAL_CONTINUATION_ROOT',str(tmp_path))
    monkeypatch.delenv('ADVERSARIAL_ATTACK_SESSION',raising=False)
    monkeypatch.setattr(workflow,'experiments_root',lambda:tmp_path)
    monkeypatch.setattr(workflow,'OpenAI',lambda **kw:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(workflow,'load_llm_profile',lambda:dict(api_key='offline',base_url='http://offline.invalid'))
    monkeypatch.setattr(workflow,'StreamingRequestArchive',lambda *args:SimpleNamespace())
    selections=[];states=[]
    def select(scene,folder,instruction,assets,constraints,*args):
        selections.append(scene);folder.mkdir(parents=True)
        times=np.arange(24)*.5
        seed=folder/'seed.csv';write_trajectory(seed,times,np.tile([30.,8.],(24,1)),np.zeros(24),[4.,2.,1.5])
        result=dict(status='completed',instruction=instruction,target=dict(operation='edit_existing',track_id='other'),
            seed_trajectory_path=str(seed),editing_scene=str(scene))
        (folder/'output.json').write_text(json.dumps(result))
        return result
    class Editor:
        def __init__(self,scene,selection,folder,constraints):
            self.run_dir=folder;folder.mkdir();self.nodes={};self.history=[]
            self.constraints=constraints;self.pdm_probe=self.policy_probe=None
            self.context=lambda:{'constructor_budget':{'calls':0,'evaluations':0}}
    def edit(editor,selection,*args,**kwargs):
        states.append(editor.constraints['_fixed_ego_history'])
        (editor.run_dir/'output.json').write_text(json.dumps(dict(status='not_submitted',stop_reason='finished_without_submission')))
        return None
    monkeypatch.setattr(workflow,'select_object',select)
    monkeypatch.setattr(workflow,'AttackEditor',Editor)
    monkeypatch.setattr(workflow,'initial_state_precheck',lambda editor:None)
    monkeypatch.setattr(workflow,'edit_object',edit)
    ledger=tmp_path/'first/probes';ledger.mkdir(parents=True)
    (ledger/'budget.json').write_text(json.dumps({'calls':2,'latest_feedback':None}))
    constraints=dict(prepare_ego_history=True,vadv2_probe_root=str(ledger))
    first=workflow.run(source,tmp_path/'first/attack','same task',constraints=constraints)
    second=workflow.run(source,tmp_path/'second/attack','same task',constraints=constraints,
        continue_from=tmp_path/'first/attack')
    assert first['ego_history_preparation']['status']=='prepared'
    assert second['ego_history_preparation']['status']=='reused'
    assert len(selections)==1 and states[0]['takeover']==states[1]['takeover']
    saved=json.loads((tmp_path/'second/attack/input.json').read_text())
    assert saved['constraints']['vadv2_probe_root']==str(ledger)
    assert json.loads((ledger/'budget.json').read_text())['calls']==2


def test_selection_schema_no_longer_edits_ego_and_history_revisions_share_switching_budget():
    from run_attack_workflow import OBJECT_TOOLS,TRAJECTORY_TOOLS
    from attack_plan_options import selection_tools
    from attack_tool_contract import trajectory_tools
    tools=selection_tools(OBJECT_TOOLS,[{'asset_id':'car'}],None,None,[])
    for item in tools:
        assert 'ego_speed_targets' not in item['function']['parameters']['properties']
    revision=next(item for item in TRAJECTORY_TOOLS if item['function']['name']=='request_ego_history_revision')
    assert set(revision['function']['parameters']['properties'])=={'target_speed_mps','reason'}
    assert trajectory_tools([revision],{},set(),set(),1,reselections_remaining=0)==[]
    assert not same_scene_constraints({'prepare_ego_history':False},{'prepare_ego_history':True})


def test_actor_agent_can_return_to_preparation_only_as_a_separate_action(source,tmp_path,monkeypatch):
    import run_attack_workflow as workflow
    import attack_policy_probe
    import decision_scene_view
    folder=tmp_path/'editor';folder.mkdir()
    constraints=dict(prepare_ego_history=True,vadv2_self_test=True,agent_decides_interaction=True,
        acceptance_policy=policy.for_new_run())
    editor=SimpleNamespace(run_dir=folder,scene=source,constraints=constraints,history=[],
        nodes={'seed_000':{'result':{'reference_screen_passed':True}}},pdm_probe=None,policy_probe=None,
        inspected=set(),submitted=None,target={'track_id':'other'})
    editor.public=lambda candidate:{'candidate_id':candidate,'reference_screen_passed':True}
    editor.context=lambda:{'constructor_budget':{'calls':0,'evaluations':0},'constraints':constraints}
    calls=[('probe_vadv2',{'candidate_id':'seed_000'}),
        ('request_ego_history_revision',{'target_speed_mps':2.,'reason':'Measured early stop; request a different starting speed'})]
    observed=[]
    class Probe:
        tested=False
        def context(self):return dict(candidates={'seed_000':{'policy_attack_success':False}} if self.tested else {},remaining=2)
        def probe(self,candidate):
            self.tested=True
            return dict(status='completed',candidate_id=candidate,overlay_path='/actual-vad.png',policy_attack_success=False)
        def require_reselection_evidence(self):assert self.tested
    monkeypatch.setattr(attack_policy_probe,'VADProbe',lambda editor:Probe())
    monkeypatch.setattr(decision_scene_view,'prepare_attack_scene',lambda *a:None)
    monkeypatch.setattr(workflow,'image_part',lambda path:{'type':'image_url','path':path})
    monkeypatch.setattr(workflow,'reasoning_effort',lambda:'low')
    monkeypatch.setattr(workflow,'decode_call',lambda n:({'id':str(n),'function':{'name':calls[n][0],'arguments':json.dumps(calls[n][1])}},*calls[n]))
    class Archive:
        def complete(self,*args,**kw):
            observed.append(kw)
            return '',len(observed)-1
    assert workflow.edit_object(editor,{'instruction':'edit the other vehicle'},Archive(),300,16384,4) is None
    assert constraints['_ego_history_revision_requested'] is True
    assert constraints['_ego_history_revision_target']==2.
    result=json.loads((folder/'output.json').read_text())
    assert result['stop_reason']=='object_reselection_requested'
    assert len(observed)==2
    prepared=json.loads((folder/'history_revision_02/report.json').read_text())
    assert prepared['status']=='prepared' and prepared['api_calls']==0
