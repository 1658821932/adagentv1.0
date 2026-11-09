"""Coarse planning must be grounded in the retrieved scene, before expensive search."""
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from tools.registry import call_tool
from tools.tool04_edit_planner.real import run

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
    assert result['result']['suitability']=='ready'
    assert arguments5['vehicle']==vehicle
    assert arguments5['maneuver']=='stationary' and 'target_track' not in arguments5
    assert call_tool('attack_vehicle_editor',arguments5,mode='mock')['status']=='success'


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
