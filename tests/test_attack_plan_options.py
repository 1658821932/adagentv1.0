"""Exercise multi-option orchestration offline, including failed object changes."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))

from attack_budget import AUTO_SESSION_ROOT, BudgetedArchive, constructor_budget_path
from attack_plan_options import choice, editor_deadline, selection_tools, shared_auto_session
import run_attack_workflow as workflow


def options():
    return [dict(option_id='existing',operation='edit_existing',target_track='lead',
                 instruction='Brake the existing lead.',rationale='Visible lead.',uncertainty='Check initial state.'),
            dict(option_id='added',operation='add_vehicle',maneuver='stationary',
                 instruction='Place a stationary lead.',rationale='Flexible placement.',uncertainty='Check road support.')]


def test_object_tools_preserve_explicit_locks_and_hide_unavailable_assets():
    original=deepcopy(workflow.OBJECT_TOOLS)
    tools=selection_tools(original,[],None,None,options())
    assert [t['function']['name'] for t in tools]==['finish_search','select_vehicle']
    assert tools[1]['function']['parameters']['properties']['option_id']['enum']==['existing','added']
    assert 'option_id' not in original[1]['function']['parameters']['properties']
    locked=selection_tools(original,[{'asset_id':'car'}],'edit_existing','lead',options())
    assert 'add_vehicle' not in [t['function']['name'] for t in locked]
    added=selection_tools(original,[{'asset_id':'car'}],'add_vehicle',None,options())
    assert 'select_vehicle' not in [t['function']['name'] for t in added]


def test_adapted_choice_reports_actual_operation_without_making_suggestion_a_contract():
    actual=dict(operation='add_vehicle',track_id='synthetic_vehicle_001')
    selected=choice(options(),dict(option_id='existing',reason='Existing initial state is unsupported.'),actual)
    assert selected['adapted'] and selected['operation']=='add_vehicle'
    assert selected['suggestion']['target_track']=='lead'
    with pytest.raises(ValueError,match='Unknown'):
        choice(options(),dict(option_id='invented',reason='fixture'),actual)


@pytest.fixture
def offline_workflow(tmp_path,monkeypatch):
    monkeypatch.setenv('ADVERSARIAL_EXPERIMENTS_ROOT',str(tmp_path))
    monkeypatch.setenv('ADVERSARIAL_CONTINUATION_ROOT',str(tmp_path))
    monkeypatch.delenv('ADVERSARIAL_ATTACK_SESSION',raising=False)
    monkeypatch.setattr(workflow,'experiments_root',lambda:tmp_path)
    scene=tmp_path/'scene';(scene/'metadata').mkdir(parents=True)
    frames=[dict(timestamp=1000000+i*500000,ego2global=np.eye(4).tolist()) for i in range(24)]
    (scene/'metadata/frames.json').write_text(json.dumps(frames))
    observed=SimpleNamespace(scene=scene,selections=[],edits=[],precheck=True,all_bad=False,switch=False)

    class Archive:
        def __init__(self,*args):pass
        def complete(self,*args,**kwargs):return '', 'offline'
    monkeypatch.setattr(workflow,'StreamingRequestArchive',Archive)
    monkeypatch.setattr(workflow,'OpenAI',lambda **kw:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(workflow,'load_llm_profile',lambda:dict(api_key='offline',base_url='http://offline.invalid'))

    class Editor:
        def __init__(self,scene,selection,output,constraints):
            self.run_dir=output;output.mkdir(parents=True)
            self.target=selection['target'];self.constraints=constraints
            self.pdm_probe=self.policy_probe=None
            self.nodes={'seed_000':{'result':{'failed_checks':['initial_motion']}}}
            budget=constructor_budget_path(output);budget.parent.mkdir(parents=True,exist_ok=True)
            if not budget.exists():budget.write_text(json.dumps(dict(calls=1,evaluations=8)))
        def context(self):return {'constructor_budget':json.loads(constructor_budget_path(self.run_dir).read_text())}
    monkeypatch.setattr(workflow,'AttackEditor',Editor)

    def select(scene,folder,instruction,assets,constraints,archive,timeout,max_tokens,required_operation,required_track):
        folder.mkdir(parents=True)
        archive.complete('choose',[],total_seconds=1)
        observed.selections.append(dict(config=deepcopy(constraints),operation=required_operation,track=required_track))
        operation='edit_existing' if len(observed.selections)==1 or observed.all_bad else 'add_vehicle'
        target=dict(operation=operation,track_id='lead' if operation=='edit_existing' else 'synthetic_vehicle_001')
        selection=dict(status='completed',target=target,plan_choice={'operation':operation})
        (folder/'output.json').write_text(json.dumps(selection))
        return selection
    monkeypatch.setattr(workflow,'select_object',select)
    monkeypatch.setattr(workflow,'initial_state_precheck',lambda editor:
        dict(reason='Unsupported initial steering',failed_checks=['initial_motion'])
        if observed.precheck and editor.target['operation']=='edit_existing' else None)

    def edit(editor,selection,archive,timeout,max_tokens,max_turns,stage_timeout_s):
        archive.complete('edit',[],total_seconds=1)
        observed.edits.append(dict(operation=editor.target['operation'],turns=max_turns,
            budget=editor.context()['constructor_budget'],deadline=editor_deadline(stage_timeout_s)))
        budget=editor.context()['constructor_budget']
        budget['calls']+=1;budget['evaluations']+=2
        constructor_budget_path(editor.run_dir).write_text(json.dumps(budget))
        call=editor.run_dir/'tool_calls/turn_01';call.mkdir(parents=True)
        (call/'result.json').write_text(json.dumps({'reason':'Try a different object'}))
        stop='object_reselection_requested' if observed.switch and len(observed.edits)==1 else 'finished'
        (editor.run_dir/'output.json').write_text(json.dumps(dict(status='not_submitted',stop_reason=stop,reason='Try a different object')))
        return None
    monkeypatch.setattr(workflow,'edit_object',edit)
    return observed


def run_auto(observed,tmp_path):
    return workflow.run(observed.scene,tmp_path/'attack','Same scene mechanism.',
        constraints={'plan_options':options()},max_turns=5)


def test_unsupported_initial_state_switches_inside_tool_and_shares_budget(offline_workflow,tmp_path):
    seen=offline_workflow;result=run_auto(seen,tmp_path)
    assert result['status']=='continued_in_object_revision'
    assert len(seen.selections)==2 and seen.edits[0]['operation']=='add_vehicle'
    assert all(row['operation'] is None and row['track'] is None for row in seen.selections)
    feedback=seen.selections[1]['config']['object_selection_feedback']
    assert feedback[0]['target_track']=='lead' and 'steering' in feedback[0]['reason']
    assert seen.edits[0]['budget']==dict(calls=1,evaluations=8)
    budget=json.loads((tmp_path/'attack/plan-session-budget/api-budget.json').read_text())
    assert budget['requests']==3
    assert AUTO_SESSION_ROOT.get() is None


def test_agent_requested_switch_keeps_remaining_actions_and_deadline(offline_workflow,tmp_path):
    seen=offline_workflow;seen.precheck=False;seen.switch=True
    result=run_auto(seen,tmp_path)
    assert result['status']=='continued_in_object_revision'
    assert [(e['operation'],e['turns']) for e in seen.edits]==[('edit_existing',5),('add_vehicle',4)]
    assert seen.edits[0]['deadline']==seen.edits[1]['deadline']
    assert seen.edits[0]['budget']==dict(calls=1,evaluations=8)
    assert seen.edits[1]['budget']==dict(calls=2,evaluations=10)
    budget=json.loads((tmp_path/'attack/plan-session-budget/api-budget.json').read_text())
    assert budget['requests']==4


def test_repeated_bad_choices_stop_without_materializing_a_scene(offline_workflow,tmp_path):
    seen=offline_workflow;seen.all_bad=True
    result=run_auto(seen,tmp_path)
    assert result['status']=='continued_in_object_revision'
    assert len(seen.selections)==3 and not seen.edits
    final=tmp_path/'attack/object_revision_01/object_revision_02/output.json'
    exhausted=json.loads(final.read_text())
    assert exhausted['status']=='not_submitted'
    assert exhausted['stop_reason']=='object_reselection_budget_exhausted'
    assert not list((tmp_path/'attack').rglob('frozen_scene'))


def test_explicit_fixed_target_does_not_switch_automatically(offline_workflow,tmp_path):
    seen=offline_workflow
    result=workflow.run(seen.scene,tmp_path/'fixed','Fixed original lead.',
        max_turns=5,required_operation='edit_existing',required_track='lead')
    assert result['status']=='not_submitted' and len(seen.selections)==1
    assert seen.selections[0]['track']=='lead' and not seen.edits


@pytest.mark.parametrize('operation',['edit_existing','add_vehicle'])
def test_real_selector_parses_option_id_and_returns_actual_choice(tmp_path,monkeypatch,operation):
    import decision_scene_view
    scene=tmp_path/'scene';(scene/'metadata').mkdir(parents=True)
    frames=[dict(timestamp=1000000+i*500000,ego2global=np.eye(4).tolist(),
        anns=dict(track_tokens=['lead'],gt_boxes=[[30,0,0,4.2,1.8,1.6,0]])) for i in range(24)]
    (scene/'metadata/frames.json').write_text(json.dumps(frames))
    actors=[[dict(track='lead',label='vehicle',x=30.,y=0.,yaw=0.,length=4.2,width=1.8)] for _ in frames]
    monkeypatch.setattr(workflow,'observed_objects_from_frames',lambda frames:actors)
    monkeypatch.setattr(workflow,'load_official_drivable',lambda path:None)
    monkeypatch.setattr(workflow,'_road_sections',lambda *args:[])
    monkeypatch.setattr(workflow,'check_drivable_area_official',lambda *args,**kw:{'passed':True})
    monkeypatch.setattr(workflow,'actor_observation_series',lambda *args:[])
    monkeypatch.setattr(workflow,'image_part',lambda path:{'type':'image_url','image_url':{'url':'offline'}})
    monkeypatch.setattr(decision_scene_view,'render_overview',lambda *args,**kw:tmp_path/'bev.png')
    monkeypatch.setattr(decision_scene_view,'actor_view',lambda *args:[])
    monkeypatch.setattr(workflow,'AttackEditor',lambda *args:SimpleNamespace(public=lambda key:{'status':'fixture'}))
    recorded=[]
    class Archive:
        def complete(self,label,messages,**kwargs):
            recorded.append(kwargs['tools'])
            arguments=dict(option_id='existing',reason='Choose or adapt the coarse suggestion.')
            if operation=='edit_existing':arguments['track_id']='lead';name='select_vehicle'
            else:
                arguments.update(asset_id='car',x_m=20.,y_m=0.,yaw_rad=0.,speed_mps=0.);name='add_vehicle'
            folder=tmp_path/'response';folder.mkdir()
            call=dict(id='call_1',type='function',function={'name':name,'arguments':json.dumps(arguments)})
            (folder/'response.json').write_text(json.dumps({'choices':[{'message':{'tool_calls':[call]}}]}))
            return '',folder
    selection=workflow.select_object(scene,tmp_path/'choice','Same braking mechanism.',
        [dict(asset_id='car',path=str(tmp_path/'car.ply'),size_m=[4.2,1.8,1.6])],
        dict(plan_options=options(),official_margin_m=.1,attack_prefix_lock_s=0.),Archive(),30,1024)
    assert selection['status']=='completed'
    assert selection['target']['operation']==operation
    assert selection['plan_choice']['option_id']=='existing'
    assert selection['plan_choice']['adapted']==(operation=='add_vehicle')
    assert len(recorded)==1


def test_auto_budget_checkpoint_reuses_ledger_and_rejects_an_outside_root(tmp_path,monkeypatch):
    monkeypatch.delenv('ADVERSARIAL_ATTACK_SESSION',raising=False)
    monkeypatch.setenv('ADVERSARIAL_CONTINUATION_ROOT',str(tmp_path))
    roots=[]
    @shared_auto_session
    def session(output,constraints,continue_from=None):
        output.mkdir()
        roots.append(AUTO_SESSION_ROOT.get())
        (output/'input.json').write_text(json.dumps({'auto_session_root':AUTO_SESSION_ROOT.get()}))
        archive=BudgetedArchive(SimpleNamespace(complete=lambda *args,**kw:('',None)),output)
        archive.complete('offline',[],total_seconds=1)
        return archive.context()['requests']
    first=tmp_path/'first';second=tmp_path/'second'
    assert session(first,{'plan_options':options()})==1
    assert session(second,{'plan_options':options()},continue_from=first)==2
    assert roots[0]==roots[1] and AUTO_SESSION_ROOT.get() is None
    (second/'input.json').write_text(json.dumps({'auto_session_root':str(tmp_path.parent/'outside')}))
    with pytest.raises(ValueError,match='outside'):
        session(tmp_path/'third',{'plan_options':options()},continue_from=second)
    assert not (tmp_path/'third').exists() and AUTO_SESSION_ROOT.get() is None
