"""A PDM overlay is delivered to the model but does not bypass trajectory inspection."""
import ast
import json
from pathlib import Path
import sys
import types

import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


def test_probe_overlay_then_trajectory_inspection_before_submit(monkeypatch,tmp_path):
    tools=['probe_pdm','inspect_trajectory','submit_candidate','finish_search','request_object_reselection','evaluate_candidate']
    calls=[('probe_pdm',dict(candidate_id='seed_000')),('inspect_trajectory',dict(candidate_id='seed_000')),
           ('submit_candidate',dict(candidate_id='seed_000',reason='tested and viewed'))]
    editor=types.SimpleNamespace(run_dir=tmp_path,constraints=dict(stationary_target=True,pdm_self_check=True),
        history=[],nodes=dict(seed_000=dict(result=dict(reference_screen_passed=True))),inspected=set(),submitted=None,target=dict(track_id='target'))
    class Probe:
        def __init__(self,owner):self.owner=owner;self.tested=False
        def context(self):return dict(candidates={'seed_000':{}} if self.tested else {},remaining=0 if self.tested else 1)
        def probe(self,candidate):
            self.tested=True
            return dict(status='completed',candidate_id=candidate,pdm_defense_succeeded=False,overlay_path='/pdm.png')
    module=types.ModuleType('attack_pdm_probe');module.PDMProbe=Probe;monkeypatch.setitem(sys.modules,'attack_pdm_probe',module)
    editor.public=lambda candidate:dict(candidate_id=candidate,reference_screen_passed=True)
    editor.context=lambda:dict(constructor_budget=dict(calls=0,evaluations=0))
    editor.inspect=lambda candidate:dict(candidate_id=candidate,path='/trajectory.png')
    def submit(candidate,reason):
        assert candidate in editor.inspected
        editor.submitted=candidate
        return dict(status='accepted')
    editor.submit=submit
    observed=[]
    class Archive:
        def complete(self,label,messages,**kwargs):
            index=len(observed);observed.append(messages)
            if index==1:
                assert not editor.inspected  # The PDM overlay alone is not trajectory inspection.
                assert messages[0]['content'][1]['path']=='/pdm.png'
            if index==2:assert messages[0]['content'][1]['path']=='/trajectory.png'
            return '',index
    def write(path,value):Path(path).write_text(json.dumps(value))
    from mcp_servers.scene_editing.core.adversarial.attack_tool_contract import constructor_exhaustion, trajectory_tools
    from mcp_servers.scene_editing.core.adversarial.model_context import compact_model_context
    namespace=dict(json=json,compact_model_context=compact_model_context,trajectory_tools=trajectory_tools,constructor_exhaustion=constructor_exhaustion,TRAJECTORY_TOOLS=[dict(function=dict(name=name,parameters=dict(properties=dict(candidate_id=dict(type='string'))))) for name in tools],
        STATIC_EDIT_PROMPT='static',EDIT_PROMPT='moving',PDM_PROBE_PROMPT='PDM self-check',
        atomic_json=write,image_part=lambda path:dict(path=path),
        decode_call=lambda index:(dict(),*calls[index]),archive_exchange=lambda *args:None)
    node=next(n for n in ast.parse((CORE/'run_attack_workflow.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='edit_object');node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'edit_object','exec'),namespace)
    result=namespace['edit_object'](editor,dict(instruction='stationary'),Archive(),300,16384,3)
    assert result=='seed_000' and len(observed)==3
    assert 'PDM self-check' in observed[0][0]['content']


@pytest.mark.parametrize('initial_remaining',[0,1])
@pytest.mark.parametrize('reselection_count',[0,1])
def test_loop_refreshes_budget_tools_and_never_forces_exhausted_constructor(tmp_path,initial_remaining,reselection_count):
    from mcp_servers.scene_editing.core.adversarial.attack_tool_contract import constructor_exhaustion, trajectory_tools
    from mcp_servers.scene_editing.core.adversarial.model_context import compact_model_context
    budget=dict(calls=0,evaluations=1000-initial_remaining,remaining_evaluations=initial_remaining)
    editor=types.SimpleNamespace(run_dir=tmp_path,constraints=dict(object_reselection_count=reselection_count),
        history=[],nodes={'seed_000':dict(result=dict(reference_screen_passed=True))},pdm_probe=None,submitted=None,target=dict(track_id='target'))
    editor.public=lambda candidate:dict(candidate_id=candidate)
    editor.context=lambda:dict(constructor_budget=dict(budget))
    def construct(args):
        assert budget['remaining_evaluations']>0
        budget.update(calls=1,evaluations=1000,remaining_evaluations=0)
        return dict(status='completed')
    editor.construct_pool=construct
    observed=[]
    class Archive:
        def complete(self,label,messages,**kwargs):
            names={tool['function']['name'] for tool in kwargs['tools']}
            assert ('request_object_reselection' in names)==(reselection_count==0)
            assert ('construct_attack_candidates' in names)==(budget['remaining_evaluations']>0)
            if kwargs['tool_choice']!='auto':
                assert kwargs['tool_choice']['function']['name'] in names
            observed.append(kwargs)
            return '',len(observed)
    def decode(attempt):
        name='construct_attack_candidates' if initial_remaining and attempt==1 else 'finish_search'
        return {},name,dict(reason='bounded search')
    def write(path,value):Path(path).write_text(json.dumps(value))
    namespace=dict(json=json,trajectory_tools=trajectory_tools,constructor_exhaustion=constructor_exhaustion,compact_model_context=compact_model_context,
        TRAJECTORY_TOOLS=[dict(function=dict(name=name)) for name in (
            'construct_attack_candidates','request_object_reselection','finish_search')],
        STATIC_EDIT_PROMPT='static',EDIT_PROMPT='moving',atomic_json=write,
        decode_call=decode,archive_exchange=lambda *args:None)
    node=next(n for n in ast.parse((CORE/'run_attack_workflow.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='edit_object');node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'edit_object','exec'),namespace)
    assert namespace['edit_object'](editor,dict(instruction='moving'),Archive(),300,16384,3) is None
    assert len(observed)==initial_remaining+1
    assert observed[-1]['tool_choice']=='auto'
    if initial_remaining:
        assert observed[0]['tool_choice']['function']['name']=='construct_attack_candidates'
    output=json.loads((tmp_path/'output.json').read_text())
    assert output['stop_reason']=='finished_without_submission'
    assert output['constructor_budget']['remaining_evaluations']==0


@pytest.mark.parametrize('initial_remaining',[0,1])
def test_exhausted_attack_without_qualified_candidate_stops_without_another_api(tmp_path,initial_remaining):
    from mcp_servers.scene_editing.core.adversarial.attack_tool_contract import constructor_exhaustion, trajectory_tools
    from mcp_servers.scene_editing.core.adversarial.model_context import compact_model_context
    budget=dict(calls=0,evaluations=1000-initial_remaining,remaining_evaluations=initial_remaining)
    bad=dict(reference_screen_passed=False,failed_checks=['road_surface'])
    editor=types.SimpleNamespace(run_dir=tmp_path,constraints={},history=[],
        nodes={'seed_000':dict(result=bad)},pdm_probe=None,submitted=None,target=dict(track_id='target'))
    editor.public=lambda candidate:dict(candidate_id=candidate)
    editor.context=lambda:dict(constructor_budget=dict(budget))
    def construct(args):
        budget.update(calls=1,evaluations=1000,remaining_evaluations=0)
        editor.nodes['candidate_001']=dict(result=dict(bad))
        return dict(status='no_feasible_goal')
    editor.construct_pool=construct
    observed=[]
    class Archive:
        def complete(self,*args,**kwargs):
            assert not observed and initial_remaining
            observed.append(kwargs)
            return '',1
    def write(path,value):Path(path).write_text(json.dumps(value))
    namespace=dict(json=json,trajectory_tools=trajectory_tools,constructor_exhaustion=constructor_exhaustion,
        compact_model_context=compact_model_context,
        TRAJECTORY_TOOLS=[dict(function=dict(name='construct_attack_candidates'))],
        STATIC_EDIT_PROMPT='static',EDIT_PROMPT='moving',atomic_json=write,
        decode_call=lambda attempt:({},'construct_attack_candidates',dict(reason='search')),
        archive_exchange=lambda *args:None)
    node=next(n for n in ast.parse((CORE/'run_attack_workflow.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='edit_object');node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'edit_object','exec'),namespace)
    # One available turn exercises exhaustion on the last allowed model action too.
    assert namespace['edit_object'](editor,dict(instruction='moving'),Archive(),300,16384,1) is None
    assert len(observed)==initial_remaining
    output=json.loads((tmp_path/'output.json').read_text())
    assert output['stop_reason']=='constructor_budget_exhausted_without_qualified_candidate'
    assert output['termination_evidence']['failed_checks']==['road_surface']
    assert output['status']=='not_submitted'
