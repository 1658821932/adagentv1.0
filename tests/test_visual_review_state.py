"""Visual eligibility, closing tools and compact evidence use actual backend helpers."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@pytest.fixture(autouse=True)
def scene_layout(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(CORE))
    import decision_scene_view
    monkeypatch.setattr(decision_scene_view,'render_overview',lambda *a,**k:Path('fixture-layout.png'))
    monkeypatch.setattr(decision_scene_view,'actor_view',lambda *a,**k:{'actors':[]})


def module(name):
    spec=importlib.util.spec_from_file_location(name,CORE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


review=module('visual_review')
compact=module('ego_context_view')
transport=module('ego_tool_transport')


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


def test_attached_batch_retains_every_candidate_without_false_delivery():
    attached=dict(candidate_id='b',batch=[dict(candidate_id='a'),dict(candidate_id='b')])
    state=review.review_state(nodes(),[],[],attached,None,3)
    assert state['submittable_candidate_ids']==['a','b']
    assert state['attached_candidate_ids']==['a','b']
    assert state['delivered_candidate_ids']==[]


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


def test_passed_created_candidate_closes_search_only_after_refinement():
    state=review.review_state(nodes(),[],[dict(candidate_id='a'),dict(candidate_id='b')],None,None,3)
    feedback=dict(decision_summary=dict(outcome='candidate_checks_passed',candidate_id='b',candidate_created=True))
    base,_=tools()
    closed=review.finalization_tools(base,state,feedback,refinement_pending=False)
    assert [item['function']['name'] for item in closed]==['submit_candidate']
    assert closed[0]['function']['parameters']['properties']['candidate_id']['enum']==['a','b']
    unchanged=review.finalization_tools(base,state,feedback,refinement_pending=True)
    assert [item['function']['name'] for item in unchanged]==[item['function']['name'] for item in base]


def test_repeated_visual_inspection_reuses_image_and_budget(tmp_path):
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fallback')),
        ('inspect_trajectory',dict(candidate_id='a')),
        ('inspect_trajectory',dict(candidate_id='a')),
        ('submit_candidate',dict(candidate_id='a',reason='reviewed'))]
    run,contexts=runner(tmp_path,events)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=5,max_inspections=1)
    assert result['selected_candidate_id']=='a'
    assert result['inspection_calls']==0 and len(result['inspections'])==1
    assert len(result['completed_image_deliveries'])==1
    assert len(list((tmp_path/'run/inspections').glob('inspection_*.json')))==1
    assert contexts[3]['last_tool_result']['cached'] is True


def test_passed_preview_creates_once_then_submits_without_more_foci(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(CORE))
    from agent_tool_view import with_decision_summary
    from speed_profile_contract import EXAMPLE,tool as speed_tool
    from ego_edit_contract import normalize_edit_call
    fake_generation(monkeypatch)
    preview=json.loads(json.dumps(EXAMPLE));preview['parent_candidate_id']='a'
    created=json.loads(json.dumps(preview));created['preview_only']=False
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fallback')),
        ('inspect_trajectory',dict(candidate_id='a')),
        ('edit_speed_profile',preview),('edit_speed_profile',created),
        ('submit_candidate',dict(candidate_id='edit_001',reason='checked refinement is suitable'))]
    run,contexts=runner(tmp_path,events)
    run.__globals__['TOOLS'].append(speed_tool())
    run.__globals__['EDIT_TOOL_NAMES'].add('edit_speed_profile')
    run.__globals__['normalize_edit_call']=normalize_edit_call
    editor_type=run.__globals__['EgoEditor'];original=editor_type.edit
    def structured_edit(self,args):
        if args.get('preview_only'):
            return with_decision_summary(dict(status='preview',feasible=True,candidate_created=False,
                task_contract_passed=True,dense_trajectory=[[i,0] for i in range(100)]),args)
        raw=original(self,args)
        raw.update(reference_screen_passed=True,task_contract_passed=True)
        return with_decision_summary(raw,args)
    editor_type.edit=structured_edit
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=7,max_inspections=1,controlled_generation=True)
    assert result['selected_candidate_id']=='edit_001' and result['model_actions']==5
    assert contexts[3]['_test_exposed_tools']==['edit_speed_profile']
    assert contexts[3]['_test_tool_choice']=='required'
    assert contexts[4]['_test_exposed_tools']==['submit_candidate']
    assert not contexts[4]['refinement_workflow']['pending']
    archived=json.loads((tmp_path/'run/tool-exchanges/turn_03/result.json').read_text())
    assert len(archived['dense_trajectory'])==100
    assert 'dense_trajectory' not in contexts[3]['last_tool_result']
    assert contexts[3]['last_tool_result']['archive_ref'].endswith('tool-exchanges/turn_03')


def test_compact_catalog_preserves_ids_failures_unknowns_and_key_numbers():
    rows=[]
    for i in range(25):
        rows.append(dict(candidate_id=str(i),status='reference_screen_failed',failed_checks=['target_clearance'],
            min_target_clearance_m=.2,max_acceleration_mps2=3.,coverage=dict(target_clearance='failed',no_cross_lines='unknown'),
            generation_curve=dict(max_acceleration_mps2=3.,actual_start_state=dict(x=1),scope='duplicate'*200)))
    context=dict(candidate_catalog=rows,current=rows[0],best=rows[1],current_edit_parent_id='0',
        latest_generated_candidate_id='2',pending_review_candidate_id='3',untried_initial_candidates=rows,
        edit_history=[dict(candidate_id=str(i),payload='x'*100) for i in range(5)])
    result=compact.compact_context(context)
    assert [row['candidate_id'] for row in result['edit_history']]==['2','3','4']
    assert result['edit_history_omitted_count']==2
    assert result['edit_history_archive']=='editor-state.json#history'
    decoded=[dict(zip(result['candidate_catalog']['columns'],r)) for r in result['candidate_catalog']['rows']]
    assert [r['candidate_id'] for r in decoded]==[str(i) for i in range(25)]
    assert all(r['failed_checks']==['target_clearance'] and r['unknown_checks']==['no_cross_lines'] for r in decoded)
    assert all(r['max_acceleration_mps2']==3 for r in decoded)
    assert set(result['active_candidate_details'])=={'0','2','3'}
    assert len(json.dumps(result)) < len(json.dumps(context))*.5
    assert rows[0]['generation_curve']['scope']=='duplicate'*200


def runner(tmp_path, events, scene_context=None, edit_status='reference_screen_passed', visual=True,
           image_tool_choice='required', tool_choice_policy='preserve'):
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
            self.times=np.array([0.,1.]);self.baseline_xy=np.array([[0.,0.],[1.,0.]])
            self.baseline_yaw=np.zeros(2);self.actors=[[],[]];self.road_official=self.road=None
        def model_context(self,*args,**kwargs):
            result=dict(scene_context or {}, current_edit_parent_id=self.current)
            if len(args)>2 and args[2] is not None:result['last_tool_result']=args[2]
            return result
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
            total_seconds=params.pop('total_seconds');params.pop('allow_tool_calls')
            payload=dict(model='test-model',messages=messages,**normalize('test-model',params))
            atomic(p/'request.json',dict(endpoint='https://example.invalid/v1',payload=payload))
            content=messages[0]['content']
            text=content if isinstance(content,str) else content[0]['text']
            contexts.append(json.JSONDecoder().raw_decode(text.split('Current geometry-informed state:\n')[1])[0])
            contexts[-1]['_test_exposed_tools']=[t['function']['name'] for t in params['tools']]
            contexts[-1]['_test_tool_choice']=params['tool_choice']
            last_text=content if isinstance(content,str) else content[-1].get('text','')
            if visual:
                contexts[-1]['_test_ends_with_native_action']=last_text.endswith(namespace['NATIVE_ACTION'])
            event=events.pop(0)
            if isinstance(event,Exception):atomic(p/'error.json',dict(request_total_seconds=total_seconds));raise event
            if isinstance(event,dict) and 'text' in event:
                finish='stop';message=dict(content=event['text'])
            else:
                name,args=event;call=dict(id=str(counter[0]),type='function',function=dict(name=name,arguments=json.dumps(args)))
                finish='tool_calls';message=dict(tool_calls=[call])
            atomic(p/'response.json',dict(choices=[dict(finish_reason=finish,message=message)]))
            atomic(p/'result.json',dict(status='received',finish_reason=finish))
            return '',p
    normalize=module('kimi_request_parameters').normalize_kimi_parameters
    base,inspection=tools()
    from mcp_servers.scene_editing.core.adversarial.model_context import compact_model_context
    from mcp_servers.scene_editing.core.adversarial.defense_refinement import refinement_state, VERSION, GUIDANCE
    namespace=dict(ROOT=tmp_path,refinement_state=refinement_state,REFINEMENT_VERSION=VERSION,REFINEMENT_GUIDANCE=GUIDANCE,Path=Path,json=json,compact_model_context=compact_model_context,os=__import__('os'),time=__import__('time'),EgoEditor=Editor,experiments_root=lambda:tmp_path,
        atomic_json=atomic,TOOLS=base,INSPECT=inspection,EDIT_TOOL_NAMES={'edit_trajectory_points'},
        normalize_edit_call=lambda name,args:args,PROMPT='Test editing prompt.',DEFAULT_TARGET_TRACK=None,
        load_llm_profile=lambda:dict(api_key='test',base_url='https://example.invalid/v1',model='test-model',
            image_tool_choice=image_tool_choice,tool_choice_policy=tool_choice_policy),
        reasoning_effort=lambda:'low',
        OpenAI=lambda **kw:types.SimpleNamespace(close=lambda:None),httpx=types.SimpleNamespace(Client=lambda **kw:None),
        StreamingRequestArchive=Archive,normalize_kimi_parameters=normalize,image_part=lambda path:dict(type='image_url',image_url=dict(url='fixture')),
        require_native_call=transport.require_native_call,
        validate_exposed_call=transport.validate_exposed_call,
        NativeToolProtocolError=transport.NativeToolProtocolError,
        protocol_rejection=transport.protocol_rejection,
        restore_protocol_feedback=transport.restore_protocol_feedback,
        review_state=review.review_state,visual_tool_contract=review.tool_contract,rejection=review.rejection,
        create_inspection=lambda editor,key,path:dict(candidate_id=key),
        _arguments=lambda call:json.loads(call['function']['arguments']),
        _summary=lambda editor,stop,actions,invalid,*rest:dict(status='submitted' if editor.submitted else 'not_submitted',
            selected_candidate_id=editor.submitted,stop_reason=stop,model_actions=actions,invalid_tool_calls=invalid))
    visual_ast=ast.parse((CORE/'run_ego_visual_editor_agent.py').read_text())
    for node in visual_ast.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in ('FOCUSED_CONTEXT','NATIVE_ACTION'):
            exec(compile(ast.Module(body=[node],type_ignores=[]), 'visual_editor_constants','exec'),namespace)
    functions=[('_tool_contract','run_ego_editor_agent.py'),('_restore_progress','run_ego_editor_agent.py'),
        ('run','run_ego_visual_editor_agent.py' if visual else 'run_ego_editor_agent.py')]
    for name,file in functions:
        node=next(n for n in ast.parse((CORE/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),str(CORE/file),'exec'),namespace)
    return namespace['run'],contexts


def fake_generation(monkeypatch):
    import types
    from defense_generation_contract import TOOLS
    class Generation:
        def __init__(self,editor,limit,**kwargs):self.editor=editor;self.limit=limit
        def remaining(self):return self.limit-self.editor.edit_count
        def context(self):return dict(remaining=self.remaining())
        def tools(self):return TOOLS if self.remaining() else []
        def execute(self,name,args):
            created=[]
            for _ in range(args['max_candidates']):
                parent=args.get('parent_candidate_id',args.get('parent_trajectory_id'))
                result=self.editor.edit(dict(mode='foci',parent_candidate_id=parent))
                if name=='construct_defense_candidates':
                    self.editor.nodes[result['candidate_id']]['operation']=dict(mode='point_edit',generation_source='constructor')
                created.append(result)
            return dict(status='candidates_generated',candidates=created)
    monkeypatch.setitem(__import__('sys').modules,'defense_generation',types.SimpleNamespace(DefenseGeneration=Generation))


def generation_events():
    return [('select_constructor_candidate',dict(candidate_id='a',reason='seed')),
        ('inspect_trajectory',dict(candidate_id='a')),
        ('optimize_defense_candidate',dict(parent_candidate_id='b',edit_window_s=[3.,11.5],max_candidates=2))]


def test_reviewed_initial_candidate_can_submit_without_foci_or_edit(tmp_path,monkeypatch):
    fake_generation(monkeypatch)
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='passing initial seed')),
        ('inspect_trajectory',dict(candidate_id='a')),
        ('submit_candidate',dict(candidate_id='a',reason='initial candidate already satisfies the task'))]
    run,contexts=runner(tmp_path,events)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=5,max_inspections=1,controlled_generation=True)
    assert result['selected_candidate_id']=='a' and result['model_actions']==3
    assert result['automatic_inspection_count']==1
    assert not contexts[2]['refinement_workflow']['pending']
    assert contexts[2]['refinement_workflow']['status']=='reviewed_initial_candidate_available'


def test_selected_initial_image_is_delivered_on_next_decision_without_an_inspection_api(tmp_path):
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fixture')),
        ('submit_candidate',dict(candidate_id='a',reason='reviewed attached initial diagram'))]
    run,contexts=runner(tmp_path,events)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=1)
    assert result['model_actions']==2 and result['selected_candidate_id']=='a'
    assert result['inspection_calls']==0 and result['automatic_inspection_count']==1
    assert contexts[1]['visual_submission_state']['attached_candidate_ids']==['a']
    assert result['completed_image_deliveries'][0]['request_turn']==2


def test_real_generation_branch_auto_delivers_all_outputs_and_allows_original_fallback(tmp_path,monkeypatch):
    fake_generation(monkeypatch)
    events=generation_events()+[('submit_candidate',dict(candidate_id='a',reason='explicit tradeoff after comparison'))]
    run,contexts=runner(tmp_path,events,scene_context=dict(comparison_guidance='Retain progress once reserve is met.'))
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=6,max_inspections=1,controlled_generation=True)
    assert result['selected_candidate_id']=='a' and result['invalid_tool_calls']==0
    assert result['automatic_inspection_count']==3 and result['inspection_calls']==0
    state=contexts[3]['visual_submission_state']
    assert state['attached_candidate_ids']==['edit_001','edit_002']
    assert contexts[3]['refinement_workflow']['pending'] is False
    assert contexts[3]['comparison_guidance']=='Retain progress once reserve is met.'
    assert 'submit_candidate' in contexts[3]['_test_exposed_tools']
    assert {row['candidate_id'] for row in result['completed_image_deliveries']}=={'a','edit_001','edit_002'}


def test_failed_generated_batch_request_resumes_with_every_image_and_identical_context(tmp_path,monkeypatch):
    fake_generation(monkeypatch)
    events=generation_events()+[RuntimeError('batch network fixture')]
    run,contexts=runner(tmp_path,events)
    with pytest.raises(RuntimeError,match='batch network fixture'):
        run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=6,max_inspections=1,controlled_generation=True)
    saved=json.loads((tmp_path/'run/summary.json').read_text())
    assert [row['candidate_id'] for row in saved['completed_image_deliveries']]==['a']
    events.append(('submit_candidate',dict(candidate_id='edit_002',reason='reviewed improvement')))
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=6,max_inspections=1,controlled_generation=True,resume=True)
    assert contexts[3]==contexts[4]
    assert result['selected_candidate_id']=='edit_002'
    assert len(result['completed_image_deliveries'])==3


def test_constructor_outputs_also_receive_images_without_counting_as_refinement(tmp_path,monkeypatch):
    fake_generation(monkeypatch)
    events=generation_events()[:2]+[
        ('construct_defense_candidates',dict(parent_trajectory_id='a',action='reduce_speed',start_time_range_s=[3.,4.],
            target_speed_mps=0.,deceleration_range_mps2=[2.,3.],end_behavior='hold',max_candidates=2)),
        ('edit_trajectory_points',dict(parent_candidate_id='a',reason='try useful refinement')),
        ('submit_candidate',dict(candidate_id='a',reason='retain reviewed fallback'))]
    run,contexts=runner(tmp_path,events)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=7,max_inspections=1,controlled_generation=True)
    assert contexts[3]['visual_submission_state']['attached_candidate_ids']==['edit_001','edit_002']
    assert not contexts[3]['refinement_workflow']['pending']
    assert contexts[4]['visual_submission_state']['attached_candidate_ids']==['edit_003']
    assert result['selected_candidate_id']=='a' and result['automatic_inspection_count']==4


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
            RuntimeError('network fixture')]
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
    assert 'submit_candidate' in contexts[2]['_test_exposed_tools']
    assert not contexts[2]['refinement_workflow']['pending']
    assert not contexts[3]['refinement_workflow']['pending']
    assert 'submit_candidate' in contexts[3]['_test_exposed_tools']
    assert result['selected_candidate_id']=='a' and result['invalid_tool_calls']==0
    assert result['refinement_workflow']['checked_changed_candidate_ids']==['edit_001']
    assert result['completed_image_deliveries'][-1]['candidate_id']=='edit_001'


def test_nonvisual_loop_corrects_text_without_executing_it(tmp_path):
    (tmp_path/'experiments').mkdir()
    events=[{'text':'{"op":"edit_trajectory_points"}'},
        ('select_constructor_candidate',dict(candidate_id='a',reason='native selection')),
        ('edit_trajectory_points',dict(parent_candidate_id='a',reason='native edit')),
        ('submit_candidate',dict(candidate_id='edit_001',reason='compared result'))]
    run,contexts=runner(tmp_path,events,visual=False)
    result=run(tmp_path,tmp_path,tmp_path/'experiments/run',max_turns=6)
    assert result['selected_candidate_id']=='edit_001' and result['invalid_tool_calls']==1
    assert result['model_actions']==4
    assert contexts[1]['_test_exposed_tools']==['select_constructor_candidate']
    assert all(context['_test_tool_choice']=='required' for context in contexts)


def test_nonvisual_loop_restores_protocol_budget_after_network_failure(tmp_path):
    (tmp_path/'experiments').mkdir()
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='native selection')),
        {'text':'I will submit a.'},RuntimeError('connection fixture')]
    run,contexts=runner(tmp_path,events,visual=False)
    directory=tmp_path/'experiments/run'
    with pytest.raises(RuntimeError,match='connection fixture'):
        run(tmp_path,tmp_path,directory,max_turns=6)
    events.append(('submit_candidate',dict(candidate_id='a',reason='native fallback')))
    result=run(tmp_path,tmp_path,directory,max_turns=6,resume=True)
    assert result['selected_candidate_id']=='a' and result['invalid_tool_calls']==1
    assert contexts[-2]==contexts[-1]


def test_text_json_is_not_executed_and_is_corrected_within_existing_budget(tmp_path):
    text={'text':'```json\n{"op":"edit_trajectory_points","preview_only":false}\n```'}
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fallback')),
        ('inspect_trajectory',dict(candidate_id='a')),text,
        ('edit_trajectory_points',dict(parent_candidate_id='a',reason='native edit after correction')),
        ('submit_candidate',dict(candidate_id='edit_001',reason='reviewed native result'))]
    run,contexts=runner(tmp_path,events)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=6,max_inspections=2)
    assert result['selected_candidate_id']=='edit_001'
    assert result['model_actions']==5 and result['invalid_tool_calls']==1
    assert result['automatic_inspection_count']==2
    assert all(context['_test_tool_choice']=='required' for context in contexts)
    assert contexts[3]['last_tool_result']['executed'] is False
    assert contexts[3]['last_tool_result']['error_code']=='native_tool_call_required'
    assert not (tmp_path/'run/tool-exchanges/turn_03').exists()
    record=json.loads((tmp_path/'run/protocol-errors/turn_03.json').read_text())
    assert record['feedback']['received_call_count']==0


def test_repeated_missing_native_calls_raise_protocol_error_not_search_refusal(tmp_path):
    events=[{'text':'I will edit now.'} for _ in range(3)]
    run,contexts=runner(tmp_path,events)
    with pytest.raises(transport.NativeToolProtocolError,match='budget exhausted'):
        run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=6,max_inspections=2)
    saved=json.loads((tmp_path/'run/summary.json').read_text())
    assert saved['stop_reason']=='native_tool_protocol_failed'
    assert saved['invalid_tool_calls']==saved['model_actions']==3
    assert saved['selected_candidate_id'] is None
    assert len(list((tmp_path/'run/protocol-errors').glob('*.json')))==3
    assert not list((tmp_path/'run/tool-exchanges').glob('*'))


def test_missing_call_on_last_turn_is_an_execution_error_even_with_remaining_invalid_budget(tmp_path):
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fallback')),
        ('inspect_trajectory',dict(candidate_id='a')),
        ('inspect_trajectory',dict(candidate_id='a')),{'text':'I choose a.'}]
    run,contexts=runner(tmp_path,events)
    with pytest.raises(transport.NativeToolProtocolError):
        run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=4,max_inspections=2)
    saved=json.loads((tmp_path/'run/summary.json').read_text())
    assert saved['stop_reason']=='native_tool_protocol_failed' and saved['invalid_tool_calls']==1


@pytest.mark.parametrize('image_tool_choice',['required','auto'])
def test_transport_failure_after_protocol_rejection_resumes_exact_state_and_budget(tmp_path,image_tool_choice):
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fallback')),
        ('inspect_trajectory',dict(candidate_id='a')),{'text':'{"op":"submit_candidate"}'},
        RuntimeError('connection fixture')]
    run,contexts=runner(tmp_path,events,image_tool_choice=image_tool_choice)
    with pytest.raises(RuntimeError,match='connection fixture'):
        run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=5,max_inspections=2)
    before=json.loads((tmp_path/'run/summary.json').read_text())
    assert before['model_actions']==3 and before['invalid_tool_calls']==1
    events.append(('submit_candidate',dict(candidate_id='a',reason='native reviewed fallback')))
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=5,max_inspections=2,resume=True)
    assert contexts[-2]==contexts[-1]
    assert result['selected_candidate_id']=='a' and result['invalid_tool_calls']==1
    assert result['model_actions']==4


@pytest.mark.parametrize('tool_choice_policy',['preserve','auto'])
def test_auto_requests_require_an_explicit_native_action_without_changing_wire_choice(tmp_path,tool_choice_policy):
    events=[('select_constructor_candidate',dict(candidate_id='a',reason='fallback')),
        ('inspect_trajectory',dict(candidate_id='a')),{'text':'{"op":"edit_trajectory_points"}'},
        ('edit_trajectory_points',dict(parent_candidate_id='a',reason='native correction')),
        ('submit_candidate',dict(candidate_id='edit_001',reason='reviewed native result'))]
    run,contexts=runner(tmp_path,events,image_tool_choice='auto',tool_choice_policy=tool_choice_policy)
    result=run(tmp_path,tmp_path,None,tmp_path/'run',max_turns=6,max_inspections=2)
    choices=['auto','auto','auto','required' if tool_choice_policy=='preserve' else 'auto','auto']
    assert [c['_test_tool_choice'] for c in contexts]==choices
    assert [c['_test_ends_with_native_action'] for c in contexts]==[choice=='auto' for choice in choices]
    assert contexts[2]['visual_submission_state']['attached_candidate_ids']==['a']
    assert contexts[4]['visual_submission_state']['attached_candidate_ids']==['edit_001']
    assert contexts[3]['last_tool_result']['executed'] is False
    assert result['selected_candidate_id']=='edit_001' and result['invalid_tool_calls']==1
