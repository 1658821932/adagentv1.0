"""Coarse planning must be grounded in the retrieved scene, before expensive search."""
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from tools.registry import call_tool
from tools.tool04_edit_planner.real import run
from mcp_servers.scene_editing.adapter.schemas import AttackInput

@pytest.fixture
def scene(tmp_path):
    root=tmp_path/'scene';(root/'metadata').mkdir(parents=True)
    (root/'asset').mkdir()
    import base64
    image=root/'frame.png'
    image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jMZkAAAAASUVORK5CYII='))
    (root/'cameras.json').write_text('{}')
    (root/'metadata/frames.json').write_text(json.dumps([dict(timestamp=1000000,anns=dict(
        track_tokens=['vehicle-A'],gt_names=['vehicle'],gt_boxes=[[0,10,0,4,2,1.5,0]],gt_velocity_3d=[[0,2,0]]),cams={"CAM_F0":{"data_path":str(image)}})]))
    (root/'metadata/map_drivable_official.json').write_text('{}')
    return dict(scene_id='source',scene_token='token',sample_root=str(root),dataset='test',matched_camera='CAM_F0',
        frame_count=1,fps=2.,manifest_path=str(root/'manifest.json'),frames_path=str(root/'metadata/frames.json'),
        cameras_path=str(root/'cameras.json'),cameras={},asset_path=str(root/'asset'),images_complete=False)

def arguments(scene):
    return dict(retrieval_result={'scene_ref':scene,'mode':'real','video_path':'/retrieved/front.mp4','match_score':.8},scene_requirement='Front vehicle slows on a straight road.',
        target_effect='Ego slows and stops safely.',edit_strategy='Modify one existing front vehicle.')

def client(response):
    return SimpleNamespace(model='offline',chat=lambda messages:json.dumps(response))

def response(track='vehicle-A'):
    return dict(suitability='ready',reasons=['The annotated front vehicle can be edited.'],
        edit_plan=dict(operation='edit_existing',target_track=track,instruction='Slow the annotated front vehicle on its existing route.'),
        retrieval_suggestion=None)

def test_ready_plan_binds_actual_scene_and_preserves_global_ego_goal(scene):
    value=arguments(scene);value['_llm_client']=client(response())
    result=call_tool('edit_planner',value,mode='real')
    assert result['status']=='success',result
    assert result['result']['scene_token']=='token'
    assert result['result']['target_effect']==value['target_effect']
    assert result['result']['edit_plan']['target_track']=='vehicle-A'
    assert result['result']['evidence']['frame_indices']==[0]

def test_missing_assets_returns_information_request_without_calling_model(scene):
    Path(scene['asset_path']).rmdir()
    def fail(messages):raise AssertionError('Do not spend a model request on missing prerequisites')
    value=arguments(scene);value['_llm_client']=SimpleNamespace(model='offline',chat=fail)
    result=run(value)
    assert result['status']=='success'
    assert result['result']['suitability']=='needs_information'
    assert result['result']['edit_plan'] is None
    assert result['meta']['attempts']==0

def test_plan_cannot_invent_an_existing_vehicle(scene):
    value=arguments(scene);value['_llm_client']=client(response('invented-ID'))
    result=run(value)
    assert result['status']=='error'
    assert result['error']['type']=='output_validation_error'

def test_unsuitable_requires_a_new_retrieval_direction(scene):
    value=arguments(scene);value['_llm_client']=client(dict(suitability='unsuitable',reasons=['No suitable road'],edit_plan=None))
    assert run(value)['error']['type']=='output_validation_error'

def test_mock_is_not_a_real_readiness_claim(scene):
    result=call_tool('edit_planner',arguments(scene),mode='mock')
    assert result['result']['suitability']=='needs_information'
    assert result['meta']['mode']=='mock'

def test_saved_retrieval_reference_is_accepted(scene,tmp_path):
    saved=tmp_path/'retrieval.json';saved.write_text(json.dumps(dict(status='success',tool='video_retriever',result=dict(scene_ref=scene))))
    value=arguments(scene);value['retrieval_result']=str(saved);value['_llm_client']=client(response())
    assert run(value)['result']['scene_id']=='source'


def test_scene_evidence_is_reused_only_for_identical_planning_input(scene,monkeypatch):
    from tools.tool04_edit_planner import real
    original=real.inspect_scene;uncached=[]
    def inspect(data):
        if real._INSPECTION.get() is None:uncached.append(data.target_effect)
        return original(data)
    monkeypatch.setattr(real,'inspect_scene',inspect)
    value=arguments(scene);value['_llm_client']=client(response())
    assert run(value)['status']=='success'
    assert uncached==[value['target_effect']]
    changed=arguments(scene);changed['target_effect']='Ego yields without an unnecessary full stop.'
    changed['_llm_client']=client(response())
    result=run(changed)
    assert result['status']=='success' and result['result']['target_effect']==changed['target_effect']
    assert uncached==[value['target_effect'],changed['target_effect']]


def test_source_keyframe_is_sent_as_image_not_as_a_json_string(scene):
    import base64
    path=Path(scene['sample_root'])/'frame.png'
    path.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jMZkAAAAASUVORK5CYII='))
    frames=Path(scene['frames_path']);data=json.loads(frames.read_text())
    data[0]['cams']={'CAM_F0':{'data_path':str(path)}};frames.write_text(json.dumps(data))
    received=[]
    def chat(messages):received.append(messages);return json.dumps(response())
    value=arguments(scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    assert run(value)['status']=='success'
    content=received[0][1]['content']
    images=[part for part in content if part['type']=='image_url']
    assert len(images)==1
    assert images[0]['image_url']['url'].startswith('data:image/jpeg;base64,')


def retrieved(scene):
    from tools.tool03_video_retriever.schemas import Output as RetrievalOutput
    return RetrievalOutput.model_validate(dict(scene_ref=scene,mode='real',video_path='/retrieved/front.mp4',
        source_file='/indexed/front.mp4',match_score=.8,source_segment={'start':2.,'end':5.})).model_dump(mode='json')


@pytest.mark.parametrize('form',['envelope','result','result_ref'])
def test_tool03_result_reaches_tool05_without_argument_reassembly(scene,tmp_path,form):
    source=retrieved(scene);envelope=dict(status='success',tool='video_retriever',result=source)
    upstream=envelope if form=='envelope' else source
    if form=='result_ref':
        file=tmp_path/'retrieval.json';file.write_text(json.dumps(envelope));upstream=str(file)
    value=arguments(scene);value.pop('retrieval_result');value['retrieval_result']=upstream;value['_llm_client']=client(response())
    review=call_tool('edit_planner',value,mode='real')
    assert review['status']=='success',review
    result=review['result']
    assert result['suitability']=='ready'
    if form=='result_ref':assert result['attack_input']['scene_ref']==upstream
    else:assert isinstance(result['attack_input']['scene_ref'],dict)
    assert result['evidence']['retrieval']['source_segment']==source['source_segment']
    assert result['evidence']['retrieval']['source_file']==source['source_file']
    # This proves direct public argument handoff; the editor is mocked, not a real search.
    attack=call_tool('attack_vehicle_editor',result['attack_input'],mode='mock')
    assert attack['status']=='success',attack
    assert attack['result']['target_track']=='vehicle-A'
    assert attack['result']['operation']=='edit_existing'
    assert attack['result']['source_scene_ref']['scene_token']==scene['scene_token']
    assert result['target_effect']==value['target_effect']


def addition():
    return dict(suitability='ready',reasons=['The base can host one stationary vehicle.'],edit_plan=dict(
        operation='add_vehicle',maneuver='stationary',instruction='Add one stationary vehicle ahead on the ego route.'),retrieval_suggestion=None)


def test_add_vehicle_without_asset_keeps_rough_plan_but_not_executable_arguments(scene):
    value=arguments(scene);value['_llm_client']=client(addition())
    result=run(value)
    assert result['status']=='success',result
    assert result['result']['suitability']=='needs_information'
    assert result['result']['edit_plan']['operation']=='add_vehicle'
    assert result['result']['attack_input'] is None


def test_add_vehicle_uses_supplied_asset_and_size_verbatim(scene,tmp_path):
    asset=tmp_path/'car.ply';asset.write_text('ply\nformat ascii 1.0\nend_header\n')
    vehicle=dict(asset_path=str(asset),size_m=[4.2,1.8,1.6])
    value=arguments(scene);value.update(vehicle=vehicle,_llm_client=client(addition()))
    result=run(value)
    assert result['status']=='success',result
    arguments5=result['result']['attack_input']
    assert AttackInput.model_validate(arguments5).vadv2_self_test is True
    assert result['result']['suitability']=='ready'
    assert arguments5['vehicle']==vehicle
    assert arguments5['maneuver']=='stationary' and 'target_track' not in arguments5
    assert call_tool('attack_vehicle_editor',arguments5,mode='mock')['status']=='success'


def test_asset_supplement_reuses_only_matching_rough_plan(scene,tmp_path):
    calls=[]
    def chat(messages):
        calls.append(messages)
        return json.dumps(addition())
    value=arguments(scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    first=run(value)
    assert first['result']['suitability']=='needs_information'
    asset=tmp_path/'car.ply';asset.write_text('ply\nformat ascii 1.0\nend_header\n')
    vehicle=dict(asset_path=str(asset),size_m=[4.2,1.8,1.6])
    supplemented=arguments(scene);supplemented.update(vehicle=vehicle,_llm_client=SimpleNamespace(model='offline',chat=chat))
    second=run(supplemented)
    assert second['status']=='success' and second['result']['suitability']=='ready'
    assert second['meta']['attempts']==0 and second['meta']['reused']
    assert second['result']['attack_input']['vehicle']==vehicle
    assert len(calls)==1
    changed=arguments(scene);changed['target_effect']='A different ego objective requires new planning.'
    changed.update(vehicle=vehicle,_llm_client=SimpleNamespace(model='offline',chat=chat))
    third=run(changed)
    assert third['status']=='success' and third['meta']['attempts']==1
    assert len(calls)==2


def test_partial_track_is_not_accepted_as_a_full_scene_edit_target(scene):
    from copy import deepcopy
    file=Path(scene['frames_path']);first=json.loads(file.read_text())[0]
    second=deepcopy(first);second['timestamp']+=500000;second['anns']['track_tokens']=['other-car']
    file.write_text(json.dumps([first,second]));scene['frame_count']=2
    value=arguments(scene);value['_llm_client']=client(response())
    result=run(value)
    assert result['status']=='error' and result['error']['type']=='output_validation_error'


def test_pedestrian_track_is_not_an_existing_vehicle_target(scene):
    file=Path(scene['frames_path']);frames=json.loads(file.read_text());frames[0]['anns']['gt_names']=['pedestrian'];file.write_text(json.dumps(frames))
    value=arguments(scene);value['_llm_client']=client(response())
    assert run(value)['error']['type']=='output_validation_error'


def test_wrong_or_failed_upstream_envelope_is_rejected_before_scene_inspection(scene):
    for tool,status in [('video_retriever','error'),('failure_analyzer','success')]:
        value=arguments(scene);value.pop('retrieval_result');value['retrieval_result']=dict(tool=tool,status=status,result=retrieved(scene))
        assert run(value)['error']['type']=='input_validation_error'


def test_mock_retrieval_cannot_become_a_real_edit_request(scene):
    value=arguments(scene);value.pop('retrieval_result');value['retrieval_result']=dict(retrieved(scene),mode='mock')
    result=run(value)
    assert result['status']=='success' and result['result']['attack_input'] is None
    assert result['result']['suitability']=='needs_information'


def test_unsuitable_scene_returns_retrieval_direction_without_attack_arguments(scene):
    value=arguments(scene);value['_llm_client']=client(dict(suitability='unsuitable',reasons=['Road layout mismatch.'],
        edit_plan=None,retrieval_suggestion='Retrieve a straight road with a same-lane front vehicle.'))
    result=run(value)
    assert result['status']=='success'
    assert result['result']['suitability']=='unsuitable' and result['result']['attack_input'] is None
    assert 'straight road' in result['result']['retrieval_suggestion']


def test_missing_source_images_returns_information_request_without_model(scene):
    (Path(scene['sample_root'])/'frame.png').unlink()
    value=arguments(scene);value['_llm_client']=client(response())
    result=run(value)
    assert result['result']['suitability']=='needs_information' and result['meta']['attempts']==0


def test_retrieval_manifest_from_another_scene_is_not_consumed(scene):
    Path(scene['manifest_path']).write_text(json.dumps(dict(scene_token='different',timestamps_us=[1000000])))
    value=arguments(scene);value['_llm_client']=client(response())
    assert run(value)['error']['type']=='scene_inspection_error'


def test_manifest_front_keyframe_is_used_without_automatic_side_camera(scene,tmp_path):
    from copy import deepcopy
    root=Path(scene['sample_root']);front=root/'front.png';side=root/'side.png'
    front.write_bytes((root/'frame.png').read_bytes());side.write_bytes(front.read_bytes())
    frames=json.loads(Path(scene['frames_path']).read_text());frames[0]['cams']['CAM_L0']={'data_path':'missing-side.png'}
    frames[0]['cams']['CAM_F0']['data_path']='missing-front.png'
    Path(scene['frames_path']).write_text(json.dumps(frames))
    Path(scene['manifest_path']).write_text(json.dumps(dict(scene_token=scene['scene_token'],timestamps_us=[1000000],cameras={
        'CAM_F0':{'frames':[{'timestamp_us':1000000,'image_path':str(front)}]},
        'CAM_L0':{'frames':[{'timestamp_us':1000000,'image_path':str(side)}]}})))
    value=arguments(scene);value['_llm_client']=client(response())
    result=run(value)
    assert result['status']=='success',result
    assert {item['camera'] for item in result['result']['evidence']['keyframes']}=={'CAM_F0'}


@pytest.mark.parametrize('mode',['real','mock'])
def test_old_scene_ref_parameter_is_rejected(scene,mode):
    value=arguments(scene)
    value.pop('retrieval_result')
    value['scene_ref']=scene
    result=call_tool('edit_planner',value,mode=mode)
    assert result['status']=='error'
    assert result['error']['type']=='input_validation_error'
    errors=result['error']['validation_errors']
    assert any(error['loc']==['scene_ref'] and error['type']=='extra_forbidden' for error in errors)
    assert any(error['loc']==['retrieval_result'] for error in errors)


def test_old_parameter_is_not_silently_ignored_when_new_parameter_exists(scene):
    value=arguments(scene);value['scene_ref']=scene
    result=run(value)
    assert result['error']['type']=='input_validation_error'


def test_bare_scene_reference_is_not_a_retrieval_result(scene):
    value=arguments(scene);value['retrieval_result']=scene
    result=run(value)
    assert result['error']['type']=='input_validation_error'


@pytest.fixture(autouse=True)
def temporary_observation_logs(tmp_path,monkeypatch):
    from tools.tool04_edit_planner import observations
    monkeypatch.setattr(observations,'LOG_ROOT',tmp_path/'observations')


@pytest.fixture
def temporal_scene(scene):
    from copy import deepcopy
    from PIL import Image
    root=Path(scene['sample_root']);original=json.loads(Path(scene['frames_path']).read_text())[0]
    frames=[]
    for index in range(24):
        frame=deepcopy(original);frame['timestamp']=1000000+index*500000;frame['cams']={}
        for number,camera in enumerate(['CAM_F0','CAM_L0','CAM_R0','CAM_B0']):
            folder=root/'original_images'/camera;folder.mkdir(parents=True,exist_ok=True)
            image=folder/f'{index:02d}.png'
            Image.new('RGB',(160,90),((index*10)%256,number*60,80)).save(image)
            frame['cams'][camera]={'data_path':str(image)}
        frames.append(frame)
    Path(scene['frames_path']).write_text(json.dumps(frames));scene['frame_count']=24
    return scene


def image_parts(messages):
    return [part for message in messages if message['role']=='user' and isinstance(message['content'],list)
            for part in message['content'] if part.get('type')=='image_url']


def test_initial_review_contains_12_front_frames_in_3_temporal_sheets(temporal_scene):
    from PIL import Image
    received=[]
    def chat(messages):received.append(messages);return json.dumps(response())
    value=arguments(temporal_scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    result=run(value)
    assert result['status']=='success',result
    assert len(received)==1 and len(image_parts(received[0]))==3
    evidence=result['result']['evidence'];visual=evidence['visual_observation']
    assert len(evidence['frame_indices'])==12 and evidence['frame_indices'][0]==0 and evidence['frame_indices'][-1]==23
    assert len(visual['sheets'])==3
    assert all(sheet['camera']=='CAM_F0' and len(sheet['frame_indices'])==4 and sheet['delivered'] for sheet in visual['sheets'])
    assert sum(sheet['available_frames'] for sheet in visual['sheets'])==12
    assert len({frame for sheet in visual['sheets'] for frame in sheet['frame_indices']})==12
    for sheet in visual['sheets']:
        with Image.open(sheet['path']) as image:assert image.size==(1280,776)
    assert {frame['camera'] for frame in evidence['keyframes']}=={'CAM_F0'}


def test_vlm_can_request_a_side_view_before_final_plan(temporal_scene):
    received=[]
    answers=[dict(action='request_views',cameras=['CAM_L0'],purpose='Check side-entry space for the target interaction.'),response()]
    def chat(messages):received.append(messages);return json.dumps(answers.pop(0))
    value=arguments(temporal_scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    result=run(value)
    assert result['status']=='success' and result['result']['suitability']=='ready',result
    assert len(received)==2 and len(image_parts(received[0]))==3 and len(image_parts(received[1]))==6
    visual=result['result']['evidence']['visual_observation']
    assert {sheet['camera'] for sheet in visual['sheets']}=={'CAM_F0','CAM_L0'}
    assert all(sheet['delivered'] for sheet in visual['sheets'])
    assert visual['view_requests'][0]['cameras']==['CAM_L0']
    assert visual['model_calls']==2
    assert call_tool('attack_vehicle_editor',result['result']['attack_input'],mode='mock')['status']=='success'


def test_nonfront_retrieval_does_not_automatically_expose_that_camera(temporal_scene):
    temporal_scene['matched_camera']='CAM_R0'
    value=arguments(temporal_scene);value['_llm_client']=client(response())
    result=run(value)
    assert result['status']=='success'
    assert {sheet['camera'] for sheet in result['result']['evidence']['visual_observation']['sheets']}=={'CAM_F0'}


def test_unknown_view_request_returns_feedback_without_fabricating_images(temporal_scene):
    received=[];answers=[dict(action='request_views',cameras=['CAM_UNKNOWN'],purpose='Need another view.'),response()]
    def chat(messages):received.append(messages);return json.dumps(answers.pop(0))
    value=arguments(temporal_scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    result=run(value)
    assert result['status']=='success'
    assert len(image_parts(received[1]))==3
    assert 'Unknown source cameras' in json.dumps(received[1])
    visual=result['result']['evidence']['visual_observation']
    assert len(visual['sheets'])==3 and visual['view_requests'][0]['status']=='invalid'


def test_repeated_view_requests_stop_at_budget_without_attack_arguments(temporal_scene):
    value=arguments(temporal_scene)
    value['_llm_client']=client(dict(action='request_views',cameras=['CAM_L0'],purpose='Need side view.'))
    result=run(value)
    assert result['status']=='success' and result['result']['suitability']=='needs_information'
    assert result['result']['attack_input'] is None
    assert result['result']['evidence']['visual_observation']['model_calls']==3


def test_missing_extra_camera_images_are_reported_as_missing(temporal_scene):
    for image in (Path(temporal_scene['sample_root'])/'original_images/CAM_B0').glob('*.png'):image.unlink()
    answers=[dict(action='request_views',cameras=['CAM_B0'],purpose='Check the rear.'),
             dict(suitability='needs_information',reasons=['Rear images are unavailable.'],edit_plan=None,retrieval_suggestion=None)]
    received=[]
    def chat(messages):received.append(messages);return json.dumps(answers.pop(0))
    value=arguments(temporal_scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    result=run(value)
    assert result['result']['suitability']=='needs_information'
    assert 'No selected source images' in json.dumps(received[1])
    assert len(image_parts(received[1]))==3


def test_partial_front_coverage_marks_missing_tiles_instead_of_repeating_frames(temporal_scene):
    from tools.tool04_edit_planner.observations import sample_indices
    indices=sample_indices(24,12)
    root=Path(temporal_scene['sample_root'])/'original_images/CAM_F0'
    for index in indices[1:]: (root/f'{index:02d}.png').unlink()
    value=arguments(temporal_scene);value['_llm_client']=client(dict(suitability='needs_information',reasons=['Need more source frames.'],edit_plan=None,retrieval_suggestion=None))
    result=run(value)
    sheets=result['result']['evidence']['visual_observation']['sheets']
    assert len(sheets)==3 and sum(sheet['available_frames'] for sheet in sheets)==1
    assert all(len(sheet['frame_indices'])==4 for sheet in sheets)


def test_short_scenes_use_only_actual_frames_without_padding_copies(scene):
    value=arguments(scene);value['_llm_client']=client(response())
    result=run(value)
    sheets=result['result']['evidence']['visual_observation']['sheets']
    assert len(sheets)==1 and sheets[0]['frame_indices']==[0] and sheets[0]['available_frames']==1


def test_original_future_motion_cannot_veto_a_coarse_plan(temporal_scene):
    from mcp_servers.scene_editing.core.adversarial.initial_motion_precheck import initial_motion_precheck
    path=Path(temporal_scene['frames_path']);frames=json.loads(path.read_text())
    frames[-1]['anns']['gt_boxes'][0][0]=-10.
    path.write_text(json.dumps(frames))
    assert not initial_motion_precheck(frames,'vehicle-A')['passed']
    value=arguments(temporal_scene);value['_llm_client']=client(response())
    result=run(value)
    assert result['status']=='success' and result['result']['suitability']=='ready'
    assert result['result']['evidence']['motion_feasibility']['status']=='deferred_to_tool05'
    assert 'initial_motion_precheck' not in result['result']['evidence']


def test_recorded_distance_is_a_hint_not_an_eligibility_gate(scene):
    path=Path(scene['frames_path']);frames=json.loads(path.read_text())
    frames[0]['anns']['gt_boxes'][0][:2]=[-100.,0.];path.write_text(json.dumps(frames))
    value=arguments(scene);value['_llm_client']=client(response())
    result=run(value)
    assert result['status']=='success' and result['result']['suitability']=='ready'
    assert result['result']['evidence']['eligible_vehicle_tracks']==['vehicle-A']


@pytest.fixture
def spatial_scene(temporal_scene,monkeypatch):
    from tools.tool04_edit_planner import spatial_evidence
    path=Path(temporal_scene['frames_path']);frames=json.loads(path.read_text())
    for frame in frames:
        frame['ego2global']=frame['lidar2ego']=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    path.write_text(json.dumps(frames));calls=[]
    image=Path(temporal_scene['sample_root'])/'frame.png'
    def read(scene,output,track_id=None,max_samples=24):
        calls.append(track_id);output=Path(output);output.mkdir(parents=True,exist_ok=True)
        report=dict(status='available',image_path=str(image),report_path=str(output/'spatial-evidence.json'),
            vehicle_catalog=[dict(track_id='vehicle-A',diagram_label='vehicl')],
            ego_summary={'duration_s':11.5},scope='original recorded geometry')
        if track_id is not None:
            report['trajectory']=dict(track_id=track_id,points=[dict(time_s=0.,observed=True,center_xy_m=[0,10]),dict(time_s=.5,observed=False)])
        (output/'spatial-evidence.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(spatial_evidence,'read',read)
    return temporal_scene,calls


def test_bev_replaces_snapshot_dump_but_preserves_camera_images_and_full_evidence(spatial_scene):
    scene,calls=spatial_scene;received=[]
    def chat(messages):received.append(messages);return json.dumps(response())
    value=arguments(scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    result=run(value);assert result['status']=='success',result
    assert calls==[None] and len(image_parts(received[0]))==4
    view=json.loads(received[0][1]['content'][0]['text'])
    assert 'snapshots' not in view and view['actors'][0]['track_id']=='vehicle-A'
    assert view['editing_capabilities']['edit_existing'].startswith('Edit a fully observed vehicle')
    assert 'initial-position/yaw-adjusted' in view['editing_capabilities']['edit_existing']
    raw=json.loads(Path(view['evidence']['full_scene_evidence_path']).read_text())
    assert len(raw['snapshots'])==3


def test_vehicle_trajectory_is_queried_on_demand_and_duplicates_reuse_evidence(spatial_scene):
    scene,calls=spatial_scene;received=[]
    query=dict(action='request_trajectory',track_id='vehicle-A',max_samples=24,purpose='Check conflict time')
    def chat(messages):
        received.append(messages)
        return json.dumps(query if len(received)<=2 else response())
    value=arguments(scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    result=run(value);assert result['status']=='success',result
    assert calls==[None,'vehicle-A']
    returned=json.loads(received[1][-1]['content'][0]['text'])
    assert returned['trajectory']['points'][1]['observed'] is False
    assert returned['evidence_id']=='T1'
    duplicate=json.loads(received[2][-1]['content'][0]['text'])
    assert duplicate['cached'] and duplicate['evidence_id']=='T1'
    assert len(received[2][-1]['content'])==1
    assert len(result['result']['evidence']['visual_observation']['trajectory_requests'])==2


def test_unknown_vehicle_query_does_not_read_another_track(spatial_scene):
    scene,calls=spatial_scene;received=[]
    def chat(messages):
        received.append(messages)
        return json.dumps(dict(action='request_trajectory',track_id='invented',purpose='Check') if len(received)==1 else response())
    value=arguments(scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    result=run(value);assert result['status']=='success',result
    assert calls==[None]
    assert 'Unknown vehicle ID' in received[1][-1]['content'][0]['text']


def test_trajectory_queries_have_a_finite_budget(spatial_scene):
    scene,calls=spatial_scene
    value=arguments(scene);value['_llm_client']=client(dict(action='request_trajectory',track_id='vehicle-A',purpose='Check'))
    result=run(value)
    assert result['status']=='success' and result['result']['suitability']=='needs_information'
    observation=result['result']['evidence']['visual_observation']
    assert len(observation['trajectory_requests'])==3 and observation['model_calls']==4
    assert calls==[None,'vehicle-A']


def option_response():
    return dict(suitability='ready',reasons=['Two possible ways to create the same scene mechanism.'],
        edit_plan=None,edit_options=[
            dict(option_id='existing',operation='edit_existing',target_track='vehicle-A',instruction='Brake this vehicle from its original pose in 1.5-7.5s.',rationale='An annotated lead is present.',uncertainty='Check supported initial motion.'),
            dict(option_id='added',operation='add_vehicle',maneuver='stationary',instruction='Place one stationary lead at a supported initial pose for a conflict in 1.5-7.5s.',rationale='Allows another initial placement.',uncertainty='Check road support and clearances.')])


def test_alternatives_reach_tool05_without_locking_operation_or_target(scene,tmp_path):
    asset=tmp_path/'vehicle.ply';asset.write_text('offline fixture asset')
    value=arguments(scene);value.update(vehicle={'asset_path':str(asset),'size_m':[4.2,1.8,1.6]},_llm_client=client(option_response()))
    result=run(value);assert result['status']=='success',result
    handoff=result['result']['attack_input']
    assert handoff['operation']=='auto'
    assert 'target_track' not in handoff and 'maneuver' not in handoff
    assert handoff['vehicle']==value['vehicle']
    assert len(handoff['plan_options'])==2
    assert 'option_id' in handoff['plan_options'][0]
    from mcp_servers.scene_editing.adapter.schemas import AttackInput
    assert AttackInput.model_validate(handoff).operation=='auto'
    mock=call_tool('attack_vehicle_editor',handoff,mode='mock')
    assert mock['status']=='success' and mock['result']['operation']=='edit_existing'
    assert mock['result']['plan_choice']['option_id']=='existing'


def test_missing_added_asset_does_not_block_existing_option(scene):
    value=arguments(scene);value['_llm_client']=client(option_response())
    result=run(value);assert result['status']=='success',result
    assert result['result']['suitability']=='ready'
    assert 'vehicle' not in result['result']['attack_input']
    assert result['result']['evidence']['option_resources']==[
        {'option_id':'existing','available':True},{'option_id':'added','available':False}]


def test_only_added_options_without_asset_keep_options_but_no_executable_handoff(scene):
    response=option_response();response['edit_options']=response['edit_options'][1:]
    value=arguments(scene);value['_llm_client']=client(response)
    result=run(value);assert result['status']=='success',result
    assert result['result']['suitability']=='needs_information'
    assert result['result']['attack_input'] is None
    assert result['result']['edit_options'][0]['option_id']=='added'


def test_added_options_asset_supplement_reuses_the_original_planning(scene,tmp_path):
    calls=[];response=option_response();response['edit_options']=response['edit_options'][1:]
    def chat(messages):calls.append(messages);return json.dumps(response)
    value=arguments(scene);value['_llm_client']=SimpleNamespace(model='offline',chat=chat)
    assert run(value)['result']['suitability']=='needs_information'
    asset=tmp_path/'car.ply';asset.write_text('offline asset')
    value['vehicle']=dict(asset_path=str(asset),size_m=[4.2,1.8,1.6])
    completed=run(value)
    assert completed['result']['suitability']=='ready' and completed['meta']['reused']
    assert completed['result']['attack_input']['operation']=='auto' and len(calls)==1


@pytest.mark.parametrize('damage',['no_options','fixed_target','fixed_maneuver','explicit_with_options','duplicate','added_without_asset'])
def test_auto_input_rejects_ambiguous_or_unexecutable_parameters(scene,damage):
    from pydantic import ValidationError
    from mcp_servers.scene_editing.adapter.schemas import AttackInput
    value=dict(scene_ref=scene,operation='auto',instruction='Construct a lead braking conflict.',plan_options=option_response()['edit_options'])
    if damage=='no_options':value['plan_options']=[]
    if damage=='fixed_target':value['target_track']='vehicle-A'
    if damage=='fixed_maneuver':value['maneuver']='stationary'
    if damage=='explicit_with_options':value['operation']='edit_existing'
    if damage=='duplicate':value['plan_options'][1]['option_id']='existing'
    if damage=='added_without_asset':value['plan_options']=value['plan_options'][1:]
    with pytest.raises(ValidationError):AttackInput.model_validate(value)


@pytest.mark.parametrize('damage',['duplicate','invented','too_many','ambiguous'])
def test_invalid_option_sets_do_not_become_attack_handoffs(scene,damage):
    response=option_response()
    if damage=='duplicate':response['edit_options'][1]['option_id']='existing'
    if damage=='invented':response['edit_options'][0]['target_track']='invented'
    if damage=='too_many':response['edit_options']*=2
    if damage=='ambiguous':response['edit_plan']=globals()['response']()['edit_plan']
    value=arguments(scene);value['_llm_client']=client(response)
    result=run(value)
    assert result['status']=='error' and result['error']['type']=='output_validation_error'
