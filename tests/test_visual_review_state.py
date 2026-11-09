"""Visual eligibility, closing tools and compact evidence use actual backend helpers."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


def module(name):
    spec=importlib.util.spec_from_file_location(name,CORE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


review=module('visual_review')
compact=module('ego_context_view')


def tools():
    def tool(name):return dict(type='function',function=dict(name=name,parameters=dict(properties=dict(candidate_id=dict(type='string')))))
    return [tool(n) for n in ['select_constructor_candidate','edit_trajectory_points','repair_candidate','submit_candidate','finish_search']],tool('inspect_trajectory')


def nodes():
    return {k:dict(result=dict(status=status),xy=[[0.,0.],[1.,0.]],yaw=[0.,0.],parent_id=None) for k,status in [('a','reference_screen_passed'),('b','reference_screen_passed'),('failed','reference_screen_failed')]}


def contract(state, **kwargs):
    base,inspection=tools()
    defaults=dict(selected='a',first_inspection=False,closing=False,final=False,edit_names={'edit_trajectory_points'})
    defaults.update(kwargs)
    return review.tool_contract(base,inspection,state,**defaults)


def test_generated_image_is_not_yet_recorded_as_delivered():
    state=review.review_state(nodes(),[],[],dict(candidate_id='b'),'b',3)
    assert state['delivered_candidate_ids']==[]
    assert state['attached_candidate_id']=='b'
    assert state['submittable_candidate_ids']==['b']  # executable after this request succeeds
    # An API failure commits no delivery; its attached image remains pending on resume.
    state2=review.review_state(nodes(),[],[],None,'b',3)
    assert state2['submittable_candidate_ids']==[]


def test_rejection_names_exact_target_and_retains_ready_alternatives():
    state=review.review_state(nodes(),[],[dict(candidate_id='a')],None,'b',3)
    error=review.rejection('b',state)
    assert error['suggested_call']['arguments']==dict(candidate_id='b')
    assert error['required_inspection_candidate_id']=='b'
    assert error['submittable_candidate_ids']==['a']


def test_budget_zero_does_not_force_inspection_of_numerical_best():
    state=review.review_state(nodes(),[dict(automatic=False)]*3,[dict(candidate_id='a')],None,None,3)
    choice,available=contract(state,closing=True)
    assert choice=='auto'
    assert 'submit_candidate' in [t['function']['name'] for t in available]
    assert not {'edit_trajectory_points','repair_candidate'} & {t['function']['name'] for t in available}
    assert state['manual_inspections_remaining']==0 and state['final_review_remaining']==1


def test_final_submit_enum_contains_only_passed_viewed_candidates():
    state=review.review_state(nodes(),[],[dict(candidate_id='a'),dict(candidate_id='failed')],None,None,3)
    choice,available=contract(state,closing=True,final=True)
    assert choice['function']['name']=='submit_candidate'
    assert len(available)==1
    assert available[0]['function']['parameters']['properties']['candidate_id']['enum']==['a']


def test_final_without_visual_qualification_finishes_instead_of_impossible_submit():
    state=review.review_state(nodes(),[],[],None,None,3)
    choice,available=contract(state,closing=True,final=True)
    assert choice['function']['name']=='finish_search'
    assert [t['function']['name'] for t in available]==['finish_search']


def test_point_only_does_not_bypass_required_edit():
    state=review.review_state(nodes(),[],[dict(candidate_id='a')],None,None,3)
    choice,_=contract(state,closing=True,final=True,require_edit=True,edit_count=0)
    assert choice['function']['name']=='finish_search'


def test_manual_and_closing_allowances_do_not_become_negative():
    records=[dict(automatic=False)]*3+[dict(automatic=False,closing_review=True),dict(automatic=True)]
    state=review.review_state(nodes(),records,[],None,None,3)
    assert state['manual_inspections_remaining']==0 and state['final_review_remaining']==0
    _,available=contract(state,closing=True)
    assert 'inspect_trajectory' not in [t['function']['name'] for t in available]


def test_compact_catalog_preserves_ids_failures_unknowns_and_key_numbers():
    rows=[]
    for i in range(25):
        rows.append(dict(candidate_id=str(i),status='reference_screen_failed',failed_checks=['target_clearance'],
            min_target_clearance_m=.2,max_acceleration_mps2=3.,coverage=dict(target_clearance='failed',no_cross_lines='unknown'),
            generation_curve=dict(max_acceleration_mps2=3.,actual_start_state=dict(x=1),scope='duplicate'*200)))
    context=dict(candidate_catalog=rows,current=rows[0],best=rows[1],current_edit_parent_id='0',
        latest_generated_candidate_id='2',pending_review_candidate_id='3',untried_initial_candidates=rows)
    result=compact.compact_context(context)
    decoded=[dict(zip(result['candidate_catalog']['columns'],r)) for r in result['candidate_catalog']['rows']]
    assert [r['candidate_id'] for r in decoded]==[str(i) for i in range(25)]
    assert all(r['failed_checks']==['target_clearance'] and r['unknown_checks']==['no_cross_lines'] for r in decoded)
    assert all(r['max_acceleration_mps2']==3 for r in decoded)
    assert set(result['active_candidate_details'])=={'0','2','3'}
    assert len(json.dumps(result)) < len(json.dumps(context))*.5
    assert rows[0]['generation_curve']['scope']=='duplicate'*200


def runner(tmp_path, events, scene_context=None, edit_status='reference_screen_passed'):
    """Execute the production visual loop with deterministic model and geometry adapters."""
    import types
    counter=[0];editors={};contexts=[]
    def atomic(path,value):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value))
    class Editor:
        def __new__(cls,input_dir,constructor,run_dir,**kwargs):
            if kwargs.get('resume'):return editors[str(run_dir)]
            obj=super().__new__(cls);editors[str(run_dir)]=obj;return obj
        def __init__(self,input_dir,constructor,run_dir,**kwargs):
            if kwargs.get('resume'):return
            self.run_dir=run_dir;run_dir.mkdir();self.nodes=nodes();self.current=None;self.submitted=None
            self.edit_count=0;self.history=[];self.effective_policy={}
        def model_context(self,*args,**kwargs):return dict(scene_context or {}, current_edit_parent_id=self.current)
        def select_constructor(self,args):self.current=args['candidate_id'];return dict(status='selected')
        def submit(self,args):self.submitted=args['candidate_id'];return dict(status='accepted')
        def edit(self,args):
            self.edit_count+=1;key=f'edit_{self.edit_count:03d}'
            self.nodes[key]=dict(result=dict(status=edit_status),
                operation=dict(mode=args['mode']),parent_id=args['parent_candidate_id'],
                xy=[[0.,0.],[2.,0.]],yaw=[0.,0.])
            return dict(status=edit_status,candidate_id=key)
    class Archive:
        def __init__(self,folder,client,profile):self.folder=folder;self.profile=profile
        def complete(self,label,messages,**params):
            counter[0]+=1;p=self.folder/f'{label}_{counter[0]}'/'attempt_001';p.mkdir(parents=True)
            params.pop('total_seconds');params.pop('allow_tool_calls')
            payload=dict(model='test-model',messages=messages,**normalize('test-model',params))
            atomic(p/'request.json',dict(endpoint='https://example.invalid/v1',payload=payload))
            text=messages[0]['content'];text=text if isinstance(text,str) else text[0]['text']
            contexts.append(json.loads(text.split('Current geometry-informed state:\n')[1].split('\nRequested diagram')[0]))
            contexts[-1]['_test_exposed_tools']=[t['function']['name'] for t in params['tools']]
            event=events.pop(0)
            if isinstance(event,Exception):atomic(p/'error.json',{});raise event
            name,args=event;call=dict(id=str(counter[0]),type='function',function=dict(name=name,arguments=json.dumps(args)))
            atomic(p/'response.json',dict(choices=[dict(finish_reason='tool_calls',message=dict(tool_calls=[call]))]))
            atomic(p/'result.json',dict(status='received',finish_reason='tool_calls'))
            return '',p
    normalize=module('kimi_request_parameters').normalize_kimi_parameters
    base,inspection=tools()
    from mcp_servers.scene_editing.core.adversarial.model_context import compact_model_context
    from mcp_servers.scene_editing.core.adversarial.defense_refinement import refinement_state, VERSION, GUIDANCE
    namespace=dict(refinement_state=refinement_state,REFINEMENT_VERSION=VERSION,REFINEMENT_GUIDANCE=GUIDANCE,Path=Path,json=json,compact_model_context=compact_model_context,os=__import__('os'),EgoEditor=Editor,experiments_root=lambda:tmp_path,
        atomic_json=atomic,TOOLS=base,INSPECT=inspection,EDIT_TOOL_NAMES={'edit_trajectory_points'},
        normalize_edit_call=lambda name,args:args,PROMPT='Test editing prompt.',DEFAULT_TARGET_TRACK=None,
        load_llm_profile=lambda:dict(api_key='test',base_url='https://example.invalid/v1',model='test-model'),
        OpenAI=lambda **kw:types.SimpleNamespace(close=lambda:None),httpx=types.SimpleNamespace(Client=lambda **kw:None),
        StreamingRequestArchive=Archive,normalize_kimi_parameters=normalize,image_part=lambda path:dict(type='image_url',image_url=dict(url='fixture')),
        constrained_auto=module('ego_tool_transport').constrained_auto,
        validate_exposed_call=module('ego_tool_transport').validate_exposed_call,
        review_state=review.review_state,visual_tool_contract=review.tool_contract,rejection=review.rejection,
        create_inspection=lambda editor,key,path:dict(candidate_id=key),
        _arguments=lambda call:json.loads(call['function']['arguments']),
        _summary=lambda editor,stop,actions,invalid,*rest:dict(status='submitted' if editor.submitted else 'not_submitted',
            selected_candidate_id=editor.submitted,stop_reason=stop,model_actions=actions,invalid_tool_calls=invalid))
    for name,file in [('_tool_contract','run_ego_editor_agent.py'),('run','run_ego_visual_editor_agent.py')]:
        node=next(n for n in ast.parse((CORE/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),str(CORE/file),'exec'),namespace)
    return namespace['run'],contexts


def test_real_loop_retains_rejected_target_across_wrong_inspection(tmp_path):
    args=lambda key:dict(candidate_id=key,reason='fixture')
    events=[('select_constructor_candidate',args('a')),('inspect_trajectory',dict(candidate_id='a')),
        ('submit_candidate',args('b')),('inspect_trajectory',dict(candidate_id='a')),
        ('inspect_trajectory',dict(candidate_id='b')),('submit_candidate',args('b'))]
    run,contexts=runner(tmp_path,events)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=8,max_inspections=3)
    assert result['selected_candidate_id']=='b' and result['invalid_tool_calls']==1
    assert contexts[3]['visual_submission_state']['pending_review_candidate_id']=='b'
    assert contexts[4]['visual_submission_state']['pending_review_candidate_id']=='b'
    assert result['pending_review_candidate_id'] is None


def test_failed_attached_request_restores_without_false_delivery(tmp_path):
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fixture')),
            ('inspect_trajectory',dict(candidate_id='a')),RuntimeError('network fixture')]
    run,contexts=runner(tmp_path,events)
    with pytest.raises(RuntimeError,match='network fixture'):
        run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1)
    saved=json.loads((tmp_path/'run/summary.json').read_text())
    assert saved['completed_image_deliveries']==[]
    events.append(('submit_candidate',dict(candidate_id='a',reason='fixture')))
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1,resume=True)
    assert result['selected_candidate_id']=='a'
    assert len(result['completed_image_deliveries'])==1


def test_failed_first_request_with_zero_actions_can_resume_identically(tmp_path):
    events=[RuntimeError('initial connection failure')]
    run,contexts=runner(tmp_path,events)
    with pytest.raises(RuntimeError,match='initial connection failure'):
        run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1)
    saved=json.loads((tmp_path/'run/summary.json').read_text())
    assert saved['model_actions']==0 and saved['completed_image_deliveries']==[]
    events.extend([('select_constructor_candidate',dict(candidate_id='a',reason='fixture')),
                   ('inspect_trajectory',dict(candidate_id='a')),
                   ('submit_candidate',dict(candidate_id='a',reason='fixture'))])
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1,resume=True)
    assert result['selected_candidate_id']=='a' and result['model_actions']==3
    assert contexts[0]==contexts[1]


def test_zero_new_edit_budget_still_selects_inspects_and_submits(tmp_path):
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='existing passing fallback')),
            ('inspect_trajectory',dict(candidate_id='a')),
            ('submit_candidate',dict(candidate_id='a',reason='reviewed existing candidate'))]
    run,contexts=runner(tmp_path,events)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1,max_edit_candidates=0)
    assert result['selected_candidate_id']=='a' and result['model_actions']==3
    assert len(result['completed_image_deliveries'])==1


@pytest.mark.parametrize('legacy', [False, True])
def test_context_view_resume_preserves_new_compression_and_legacy_payload(tmp_path, legacy):
    scene_context = dict(other_actor_snapshots=[dict(time_s=i*.500013, track='target',
        x_m=1.234567890123456+i, scope='Recorded samples only. ' * 20) for i in range(24)])
    events=[RuntimeError('connection fixture')]
    run,contexts=runner(tmp_path,events,scene_context)
    actual_compact=run.__globals__['compact_model_context']
    if legacy:
        run.__globals__['compact_model_context']=lambda value:value
    with pytest.raises(RuntimeError,match='connection fixture'):
        run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1)
    if legacy:
        config_path=tmp_path/'run/config.json'
        config=json.loads(config_path.read_text());config.pop('context_view_version')
        config_path.write_text(json.dumps(config))
    run.__globals__['compact_model_context']=actual_compact
    events.extend([('select_constructor_candidate',dict(candidate_id='a',reason='existing candidate')),
                   ('inspect_trajectory',dict(candidate_id='a')),
                   ('submit_candidate',dict(candidate_id='a',reason='reviewed'))])
    if legacy:
        with pytest.raises(ValueError,match='context view version'):
            run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1,resume=True)
        return
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1,resume=True)
    assert result['selected_candidate_id']=='a' and contexts[0]==contexts[1]
    assert ('context_encoding' in contexts[1]) is not legacy
    assert ('context_view_version' in json.loads((tmp_path/'run/config.json').read_text())) is not legacy


def test_structured_generation_is_routed_and_full_goal_reaches_agent(monkeypatch,tmp_path):
    import types,sys
    from mcp_servers.scene_editing.core.adversarial import defense_generation_contract as contract_module
    generated=[]
    class Generation:
        def __init__(self,editor,limit,**kwargs):self.editor=editor;self.limit=limit
        def remaining(self):return self.limit-self.editor.edit_count
        def context(self):return dict(shared_new_candidates_remaining=self.remaining())
        def tools(self):return contract_module.TOOLS if self.remaining() else []
        def execute(self,name,args):
            contract_module.validate_request(name,args,boundary=3.,end=11.5,remaining=self.remaining())
            generated.append((name,args));self.editor.edit_count+=1
            cid=f'edit_{self.editor.edit_count:03d}'
            parent=args.get('parent_candidate_id','a')
            self.editor.nodes[cid]=dict(result=dict(status='reference_screen_passed'),parent_id=parent,
                operation=dict(mode='foci' if name=='optimize_defense_candidate' else 'point_edit',
                    generation_source='foci' if name=='optimize_defense_candidate' else 'constructor'),
                xy=[[0.,0.],[1.5 if name=='construct_defense_candidates' else 1.7,0.]],yaw=[0.,0.])
            return dict(status='candidates_generated',candidates=[dict(candidate_id=cid)])
    module=types.ModuleType('defense_generation');module.DefenseGeneration=Generation
    monkeypatch.setitem(sys.modules,'defense_generation',module)
    monkeypatch.setitem(sys.modules,'defense_generation_contract',contract_module)
    from mcp_servers.scene_editing.core.adversarial import defense_candidate_view as candidate_view_module
    monkeypatch.setitem(sys.modules,'defense_candidate_view',candidate_view_module)
    construction=dict(parent_trajectory_id='recorded_baseline',action='reduce_speed',start_time_range_s=[4.,4.5],
        target_speed_mps=2.,deceleration_range_mps2=[1.,3.],end_behavior='cruise',max_candidates=1)
    optimization=dict(parent_candidate_id='edit_001',edit_window_s=[4.,10.],max_candidates=1)
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='initial parent')),
        ('inspect_trajectory',dict(candidate_id='a')),('construct_defense_candidates',construction),
        ('optimize_defense_candidate',optimization),('edit_trajectory_points',dict(parent_candidate_id='edit_002',points=[],reason='refine generated seed')),
        ('submit_candidate',dict(candidate_id='edit_003',reason='reviewed refined candidate'))]
    run,contexts=runner(tmp_path,events,scene_context=dict(best=dict(candidate_id='b'),pool_preferred_candidate_id='b'))
    goal='Avoid the target; compare an evidenced moving alternative with a stop.'
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=8,max_inspections=3,
               controlled_generation=True,target_effect=goal)
    assert all('best' not in context and 'pool_preferred_candidate_id' not in context for context in contexts)
    assert all('comparison_guidance' in context for context in contexts)
    assert result['candidate_view_version']=='unranked_motion.v2'
    assert json.loads((tmp_path/'run/config.json').read_text())['version']=='ego_visual_editor.v21'
    assert result['selected_candidate_id']=='edit_003' and result['invalid_tool_calls']==0
    assert [name for name,_ in generated]==['construct_defense_candidates','optimize_defense_candidate']
    assert all(c['target_effect']==goal and 'generation_budget' in c for c in contexts)
    assert result['generation_budget']['shared_new_candidates_remaining']==2


@pytest.mark.parametrize('edit_status',['reference_screen_passed','reference_screen_failed'])
def test_new_workflow_edits_then_can_submit_original_fallback(monkeypatch,tmp_path,edit_status):
    import sys,types
    from mcp_servers.scene_editing.core.adversarial import defense_generation_contract,defense_candidate_view
    class Generation:
        def __init__(self,editor,limit,**kw):self.editor=editor;self.limit=limit
        def remaining(self):return self.limit-self.editor.edit_count
        def context(self):return dict(shared_new_candidates_remaining=self.remaining())
        def tools(self):return []
    generation=types.ModuleType('defense_generation');generation.DefenseGeneration=Generation
    monkeypatch.setitem(sys.modules,'defense_generation',generation)
    monkeypatch.setitem(sys.modules,'defense_generation_contract',defense_generation_contract)
    monkeypatch.setitem(sys.modules,'defense_candidate_view',defense_candidate_view)
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='retained fallback')),
        ('inspect_trajectory',dict(candidate_id='a')),
        ('edit_trajectory_points',dict(parent_candidate_id='b',reason='improve another source',points=[])),
        ('submit_candidate',dict(candidate_id='a',reason='compared actual edit; fallback is preferable'))]
    run,contexts=runner(tmp_path,events,edit_status=edit_status)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=6,max_inspections=2,controlled_generation=True)
    assert 'submit_candidate' not in contexts[2]['_test_exposed_tools']
    assert contexts[2]['refinement_workflow']['pending']
    assert not contexts[3]['refinement_workflow']['pending']
    assert 'submit_candidate' in contexts[3]['_test_exposed_tools']
    assert result['selected_candidate_id']=='a' and result['invalid_tool_calls']==0
    assert result['refinement_workflow']['checked_changed_candidate_ids']==['edit_001']
    assert result['completed_image_deliveries'][-1]['candidate_id']=='edit_001'
