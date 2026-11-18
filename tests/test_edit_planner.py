"""Tool04 reviews structure and adapts the tool02 plan; every case is offline."""
from copy import deepcopy
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.registry import call_tool
from tools.tool04_edit_planner.real import run

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jMZkAAAAASUVORK5CYII=')
ATTACK = dict(map_context='t_junction', ego_action='turn_right',
              other_origin='right_arm', other_action='turn_left')
ATTACK_NONE = dict(map_context='curve', ego_action='straight',
                   other_origin='none', other_action='none')


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    """Any undeclared model call fails loudly instead of reaching the network."""
    def blocked(*args, **kwargs):
        raise AssertionError('offline test: inject _llm_client instead of constructing LLMClient')
    monkeypatch.setattr('agent.llm_client.LLMClient', blocked)


@pytest.fixture(autouse=True)
def temporary_observation_logs(tmp_path, monkeypatch):
    from tools.tool04_edit_planner import observations
    monkeypatch.setattr(observations, 'LOG_ROOT', tmp_path / 'observations')


@pytest.fixture
def scene(tmp_path):
    root = tmp_path / 'scene'
    (root / 'metadata').mkdir(parents=True)
    (root / 'asset').mkdir()
    (root / 'frame.png').write_bytes(PNG)
    (root / 'cameras.json').write_text('{}')
    (root / 'metadata/frames.json').write_text(json.dumps([dict(
        timestamp=1000000,
        anns=dict(track_tokens=['vehicle-A'], gt_names=['vehicle'],
                  gt_boxes=[[0, 10, 0, 4, 2, 1.5, 0]], gt_velocity_3d=[[0, 2, 0]]),
        cams={'CAM_F0': {'data_path': str(root / 'frame.png')}})]))
    (root / 'metadata/map_drivable_official.json').write_text('{}')
    return dict(scene_id='source', scene_token='token', sample_root=str(root), dataset='test',
                matched_camera='CAM_F0', frame_count=1, fps=2.,
                manifest_path=str(root / 'manifest.json'),
                frames_path=str(root / 'metadata/frames.json'),
                cameras_path=str(root / 'cameras.json'), cameras={},
                asset_path=str(root / 'asset'), images_complete=False)


def scenario(attack=None):
    from tools.tool02_scene_instance_generator.schemas import Output as ScenarioOutput
    current = attack or ATTACK
    no_other = current.get('other_origin') == 'none'
    candidates = [dict(candidate_id=f'cand_{index}',
                       base_video_requirement=f'目标道路条件 {index}',
                       attack_edit_strategy=None if no_other else '调整右侧支路车辆通过时机',
                       ego_input_edit_strategy=None,
                       ego_correct_trajectory_strategy='自车减速后安全通过')
                  for index in (1, 2, 3)]
    return ScenarioOutput.model_validate(dict(
        target_scene=('多云白天，弯道，自车直行，无其他车辆。' if no_other
                      else '多云白天，丁字路口，自车右转，右侧支路一辆车左转。'),
        attack=current, candidates=candidates)).model_dump(mode='json')


def retrieved(scene, candidate_id='cand_1'):
    from tools.tool03_video_retriever.schemas import Output as RetrievalOutput
    return RetrievalOutput.model_validate(dict(
        scene_ref=scene, mode='real', video_path='/retrieved/front.mp4',
        source_file='/indexed/front.mp4', match_score=.8,
        source_segment={'start': 2., 'end': 5.}, candidate_id=candidate_id)).model_dump(mode='json')


def arguments(scene, candidate_id='cand_1', attack=None):
    return dict(scenario_result=scenario(attack), candidate_id=candidate_id,
                retrieval_result=retrieved(scene, candidate_id))


def client(answer):
    return SimpleNamespace(model='offline', chat=lambda messages: json.dumps(answer))


def review(suitability='ready', checks=('pass', 'pass', 'pass'), plan=True,
           reasons=('车道与空间满足目标场景。',), suggestion=None):
    value = dict(suitability=suitability, reasons=list(reasons),
                 checks=dict(zip(('road', 'lanes', 'space'), checks)),
                 retrieval_suggestion=suggestion)
    if plan:
        value['adapted_plan'] = dict(attack_edit_strategy='调整支路车辆通过时机',
                                     ego_input_edit_strategy=None,
                                     ego_correct_trajectory_strategy='自车减速让行后右转')
    else:
        value['adapted_plan'] = None
    return value


def test_ready_copies_target_scene_and_attack_by_code(scene):
    value = arguments(scene)
    value['_llm_client'] = client(review())
    result = call_tool('edit_planner', value, mode='real')
    assert result['status'] == 'success', result
    output = result['result']
    assert set(output) == set(__import__('tools.tool04_edit_planner.schemas', fromlist=['Output']).Output.model_fields)
    assert output['target_scene'] == scenario()['target_scene']
    assert output['attack'] == ATTACK
    assert output['candidate_id'] == 'cand_1'
    assert (output['scene_id'], output['scene_token']) == ('source', 'token')
    assert output['suitability'] == 'ready'
    assert output['checks'] == {'road': 'pass', 'lanes': 'pass', 'space': 'pass'}
    assert output['adapted_plan']['ego_correct_trajectory_strategy'] == '自车减速让行后右转'
    assert 'edit_plan' not in output and 'attack_input' not in output and 'target_effect' not in output


def test_ready_requires_all_checks_passing_and_a_plan(scene):
    for answer in (review(checks=('pass', 'unknown', 'pass')), review(plan=False)):
        value = arguments(scene)
        value['_llm_client'] = client(answer)
        result = run(value)
        assert result['error']['type'] == 'output_validation_error', result


def test_unsuitable_needs_a_failure_and_a_retrieval_suggestion(scene):
    legal = review(suitability='unsuitable', checks=('fail', 'pass', 'pass'), plan=False,
                   suggestion='检索一条同车道有前车的直路。')
    value = arguments(scene)
    value['_llm_client'] = client(legal)
    result = run(value)
    assert result['result']['suitability'] == 'unsuitable'
    assert result['result']['adapted_plan'] is None
    for answer in (review(suitability='unsuitable', checks=('pass', 'pass', 'pass'), plan=False,
                          suggestion='再检索'),
                   review(suitability='unsuitable', checks=('fail', 'pass', 'pass'), plan=False)):
        value = arguments(scene)
        value['_llm_client'] = client(answer)
        assert run(value)['error']['type'] == 'output_validation_error'


def test_needs_information_cannot_claim_a_confirmed_failure(scene):
    value = arguments(scene)
    value['_llm_client'] = client(review(suitability='needs_information',
                                         checks=('fail', 'unknown', 'unknown'), plan=False))
    assert run(value)['error']['type'] == 'output_validation_error'


def test_no_target_vehicle_forbids_an_attack_edit_but_allows_ego_input_editing(scene):
    blocked = review()
    value = arguments(scene, attack=ATTACK_NONE)
    value['_llm_client'] = client(blocked)
    assert run(value)['error']['type'] == 'output_validation_error'

    allowed = review()
    allowed['adapted_plan'] = dict(attack_edit_strategy=None,
                                   ego_input_edit_strategy='把自车入弯速度提前提高',
                                   ego_correct_trajectory_strategy='自车减速后按车道通过弯道')
    value = arguments(scene, attack=ATTACK_NONE)
    value['_llm_client'] = client(allowed)
    result = run(value)
    assert result['status'] == 'success', result
    assert result['result']['attack'] == ATTACK_NONE
    assert result['result']['adapted_plan']['attack_edit_strategy'] is None
    assert result['result']['adapted_plan']['ego_input_edit_strategy']


def test_model_cannot_rewrite_the_bound_values(scene):
    answer = dict(review(), target_scene='另一条路', attack=ATTACK_NONE)
    value = arguments(scene)
    value['_llm_client'] = client(answer)
    result = run(value)
    assert result['error']['type'] == 'output_validation_error'


@pytest.mark.parametrize('form', ['envelope', 'result', 'result_ref'])
def test_scenario_and_retrieval_accept_three_forms(scene, tmp_path, form):
    upstream = {'envelope': dict(status='success', tool='scene_instance_generator', result=scenario()),
                'result': scenario()}
    retrieval = dict(status='success', tool='video_retriever', result=retrieved(scene))
    if form == 'result_ref':
        scenario_file = tmp_path / 'scenario.json'
        scenario_file.write_text(json.dumps(dict(status='success', tool='scene_instance_generator',
                                                result=scenario())))
        retrieval_file = tmp_path / 'retrieval.json'
        retrieval_file.write_text(json.dumps(retrieval))
        value = dict(scenario_result=str(scenario_file), candidate_id='cand_1',
                     retrieval_result=str(retrieval_file))
    else:
        value = dict(scenario_result=upstream[form], candidate_id='cand_1',
                     retrieval_result=retrieval if form != 'result' else retrieval['result'])
    value['_llm_client'] = client(review())
    result = call_tool('edit_planner', value, mode='real')
    assert result['status'] == 'success', result
    output = result['result']
    assert (output['scene_id'], output['scene_token']) == ('source', 'token')
    assert output['evidence']['retrieval']['candidate_id'] == 'cand_1'
    assert output['evidence']['retrieval']['source_file'] == '/indexed/front.mp4'


def test_wrong_upstream_source_is_rejected(scene):
    for field, payload in (
            ('scenario_result', dict(status='success', tool='video_retriever', result=scenario())),
            ('retrieval_result', dict(status='error', tool='video_retriever', result=retrieved(scene))),
            ('scenario_result', dict(status='success', tool='scene_instance_generator',
                                     result=dict(scenario(), candidates=scenario()['candidates'][:2])))):
        value = arguments(scene)
        value[field] = payload
        assert run(value)['error']['type'] == 'input_validation_error', (field, payload)


def test_candidate_must_exist_and_match_the_retrieval(scene):
    value = arguments(scene, candidate_id='cand_9')
    assert run(value)['error']['type'] == 'input_validation_error'
    value = arguments(scene)
    value['retrieval_result'] = retrieved(scene, candidate_id='cand_2')
    assert run(value)['error']['type'] == 'input_validation_error'
    value = arguments(scene)
    value['retrieval_result'].pop('candidate_id')
    assert run(value)['error']['type'] == 'input_validation_error'


def test_legacy_input_fields_are_rejected(scene):
    for extra in ('scene_requirement', 'target_effect', 'edit_strategy', 'interaction',
                  'vehicle', 'retrieval_result_ref'):
        value = arguments(scene)
        value[extra] = 'legacy'
        result = run(value)
        assert result['error']['type'] == 'input_validation_error', extra


def test_missing_source_images_return_information_without_a_model(scene, monkeypatch):
    Path(scene['sample_root'], 'frame.png').unlink()
    clock = iter((200.0, 200.3))
    monkeypatch.setattr('tools.tool04_edit_planner.real.perf_counter', lambda: next(clock))
    result = run(arguments(scene))           # no _llm_client: the guard fails on any model call
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'needs_information'
    assert output['adapted_plan'] is None
    assert output['checks'] == {'road': 'unknown', 'lanes': 'unknown', 'space': 'unknown'}
    assert result['meta']['attempts'] == 0
    assert result['meta']['elapsed_seconds'] == 0.3


def test_missing_render_asset_does_not_block_the_structural_review(scene):
    scene['asset_path'] = str(Path(scene['sample_root']) / 'no-asset')
    value = arguments(scene)
    value['_llm_client'] = client(review())
    result = run(value)
    assert result['status'] == 'success', result
    assert result['result']['suitability'] == 'ready'
    assert result['result']['evidence']['gaussian_assets'] == []
    assert result['result']['evidence']['missing_resources'] == []


def test_mock_claims_no_inspection(scene):
    result = call_tool('edit_planner', arguments(scene), mode='mock')
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'needs_information'
    assert output['checks'] == {'road': 'unknown', 'lanes': 'unknown', 'space': 'unknown'}
    assert output['adapted_plan'] is None
    assert output['target_scene'] == scenario()['target_scene']
    assert result['meta']['mode'] == 'mock'


def test_evidence_keeps_the_review_inputs(scene):
    value = arguments(scene)
    value['_llm_client'] = client(review())
    result = run(value)
    evidence = result['result']['evidence']
    assert evidence['eligible_vehicle_tracks'] == ['vehicle-A']
    assert evidence['retrieval']['candidate_id'] == 'cand_1'
    assert evidence['frame_indices'] == [0]
    assert evidence['keyframes'][0]['camera'] == 'CAM_F0'


# --- restored visual-review regressions (new response contract) ---

# --- restored visual-review regressions (new response contract) ---




# --- data-first map pre-check (raw nuPlan map via mapfacts.collect) ---

def stub_map(monkeypatch, checks, facts=None, available=True, reasons=("stub measured map verdict",),
             suggestion=None):
    from tools.tool04_edit_planner import mapfacts
    monkeypatch.setattr(mapfacts, 'collect', lambda frames, attack: dict(
        available=available, checks=dict(checks), facts=facts if facts is not None else {'stub': True},
        reasons=list(reasons), evidence_limits=[], retrieval_suggestion=suggestion))


def test_trusted_map_failure_returns_unsuitable_without_any_model(scene, monkeypatch):
    """Scheduling test only: the current sample cannot be asserted to fail from data."""
    stub_map(monkeypatch, {'road': 'fail', 'lanes': 'unknown', 'space': 'unknown'},
             suggestion='检索具有三臂路口的场景。')
    clock = iter((100.0, 101.25))
    monkeypatch.setattr('tools.tool04_edit_planner.real.perf_counter', lambda: next(clock))
    result = run(arguments(scene))               # no _llm_client: the guard forbids any model call
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'unsuitable'
    assert output['adapted_plan'] is None
    assert output['checks']['road'] == 'fail'
    assert output['retrieval_suggestion'] == '检索具有三臂路口的场景。'
    assert output['evidence']['verification_source'] == 'measured_map_precheck'
    assert output['evidence']['map_facts'] == {'stub': True}
    assert output['evidence']['map_available'] is True
    assert result['meta']['attempts'] == 0
    assert result['meta']['elapsed_seconds'] == 1.25


def test_unavailable_map_reaches_the_visual_review(scene, monkeypatch):
    from tools.tool04_edit_planner import mapfacts
    monkeypatch.setattr(mapfacts, 'collect', lambda frames, attack: mapfacts.unavailable('raw map missing'))
    value = arguments(scene)
    value['_llm_client'] = client(review())
    result = run(value)
    assert result['status'] == 'success', result
    assert result['result']['suitability'] == 'ready'
    assert result['result']['evidence']['map_available'] is False
    assert result['result']['evidence']['map_facts'] is None


def test_visual_review_that_contradicts_a_measured_pass_is_not_accepted(scene, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'unknown', 'space': 'unknown'})
    value = arguments(scene)
    value['_llm_client'] = client(review(suitability='unsuitable', checks=('fail', 'unknown', 'unknown'),
                                         plan=False, suggestion='检索有同车道前车的连续道路。'))
    result = run(value)
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'unsuitable'      # conservative, never ready
    assert output['adapted_plan'] is None
    assert output['evidence']['precheck_conflicts'] == {'road': {'data': 'pass', 'review': 'fail'}}
    assert any('disagrees with the measured pre-check' in reason for reason in output['reasons'])
    assert output['evidence']['visual_review']['sheet_count'] >= 1


def test_visual_review_agreeing_with_map_checks_can_be_ready(scene, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
    value = arguments(scene)
    value['_llm_client'] = client(review())
    result = run(value)
    assert result['status'] == 'success', result
    assert result['result']['suitability'] == 'ready'
    assert result['result']['adapted_plan'] is not None
    assert result['result']['evidence']['verification_source'] == 'visual_review'
    assert 'precheck_conflicts' not in result['result']['evidence']


def test_raw_map_reader_returns_measured_facts_for_a_real_sample():
    """Read-only check against the real nuPlan map; skipped when the simulator env is absent."""
    from tools.tool04_edit_planner import mapfacts
    if mapfacts.python_path() is None:
        pytest.skip('simulator interpreter is not available')
    frames_path = Path('data/navtrain_ep_per1/samples/5863c4eb96825e85/metadata/frames.json')
    if not frames_path.is_file():
        pytest.skip('sample scene is not available')
    result = mapfacts.collect(json.loads(frames_path.read_text()),
                              dict(map_context='t_junction', ego_action='turn_right',
                                   other_origin='right_arm', other_action='turn_left'))
    assert result['available'] is True, result
    facts = result['facts']
    assert set(result['checks']) == {'road', 'lanes', 'space'}
    assert set(result['checks'].values()) <= {'pass', 'fail', 'unknown'}
    assert facts['approach_lane_count'] > 0
    assert facts['connector_count'] > 0
    assert 'lane_count_by_group' in facts and 'target_junction' in facts
    assert facts['selection_policy'] == 'first_real_junction_on_recorded_route'
    assert facts['map_location'] == 'us-ma-boston'
    assert 'polygon fragment' not in json.dumps(facts)     # fragment counts are not claimed as lanes


# --- restored visual-review regressions (new response contract) ---

@pytest.fixture
def temporal_scene(scene):
    from copy import deepcopy
    from PIL import Image
    root = Path(scene['sample_root'])
    original = json.loads(Path(scene['frames_path']).read_text())[0]
    frames = []
    for index in range(24):
        frame = deepcopy(original)
        frame['timestamp'] = 1000000 + index * 500000
        frame['cams'] = {}
        for number, camera in enumerate(['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0']):
            folder = root / 'original_images' / camera
            folder.mkdir(parents=True, exist_ok=True)
            image = folder / f'{index:02d}.png'
            Image.new('RGB', (160, 90), ((index * 10) % 256, number * 60, 80)).save(image)
            frame['cams'][camera] = {'data_path': str(image)}
        frames.append(frame)
    Path(scene['frames_path']).write_text(json.dumps(frames))
    scene['frame_count'] = 24
    return scene


def image_parts(messages):
    return [part for message in messages if message['role'] == 'user' and isinstance(message['content'], list)
            for part in message['content'] if part.get('type') == 'image_url']


def test_initial_review_sends_twelve_front_frames_in_three_sheets(temporal_scene):
    from PIL import Image
    received = []

    def chat(messages):
        received.append(messages)
        return json.dumps(review())

    value = arguments(temporal_scene)
    value['_llm_client'] = SimpleNamespace(model='offline', chat=chat)
    result = run(value)
    assert result['status'] == 'success', result
    assert len(received) == 1 and len(image_parts(received[0])) == 3
    evidence = result['result']['evidence']
    visual = evidence['visual_observation']
    assert len(evidence['frame_indices']) == 12
    assert evidence['frame_indices'][0] == 0 and evidence['frame_indices'][-1] == 23
    assert len(visual['sheets']) == 3
    assert all(sheet['camera'] == 'CAM_F0' and len(sheet['frame_indices']) == 4 and sheet['delivered']
               for sheet in visual['sheets'])
    assert sum(sheet['available_frames'] for sheet in visual['sheets']) == 12
    assert len({frame for sheet in visual['sheets'] for frame in sheet['frame_indices']}) == 12
    for sheet in visual['sheets']:
        with Image.open(sheet['path']) as image:
            assert image.size == (1280, 776)


def test_vlm_can_request_a_side_view_before_the_final_review(temporal_scene):
    received = []
    answers = [dict(action='request_views', cameras=['CAM_L0'],
                    purpose='Check the side entry space for the target scene.'), review()]

    def chat(messages):
        received.append(messages)
        return json.dumps(answers.pop(0))

    value = arguments(temporal_scene)
    value['_llm_client'] = SimpleNamespace(model='offline', chat=chat)
    result = run(value)
    assert result['status'] == 'success' and result['result']['suitability'] == 'ready', result
    assert len(received) == 2
    assert len(image_parts(received[0])) == 3 and len(image_parts(received[1])) == 6
    visual = result['result']['evidence']['visual_observation']
    assert {sheet['camera'] for sheet in visual['sheets']} == {'CAM_F0', 'CAM_L0'}
    assert all(sheet['delivered'] for sheet in visual['sheets'])
    assert visual['view_requests'][0]['cameras'] == ['CAM_L0']
    assert visual['model_calls'] == 2
    assert result['result']['adapted_plan'] is not None


def test_repeated_view_requests_stop_at_the_budget_without_a_plan(temporal_scene):
    value = arguments(temporal_scene)
    value['_llm_client'] = client(dict(action='request_views', cameras=['CAM_L0'],
                                       purpose='Need the side view.'))
    result = run(value)
    assert result['status'] == 'success'
    output = result['result']
    assert output['suitability'] == 'needs_information'
    assert output['adapted_plan'] is None
    assert output['checks'] == {'road': 'unknown', 'lanes': 'unknown', 'space': 'unknown'}
    assert output['evidence']['visual_observation']['model_calls'] == 3


def test_missing_extra_camera_images_are_reported_not_fabricated(temporal_scene):
    for image in (Path(temporal_scene['sample_root']) / 'original_images/CAM_B0').glob('*.png'):
        image.unlink()
    answers = [dict(action='request_views', cameras=['CAM_B0'], purpose='Check the rear.'),
               dict(suitability='needs_information', reasons=['Rear images are unavailable.'],
                    checks=dict(road='unknown', lanes='unknown', space='unknown'),
                    adapted_plan=None, retrieval_suggestion=None)]
    received = []

    def chat(messages):
        received.append(messages)
        return json.dumps(answers.pop(0))

    value = arguments(temporal_scene)
    value['_llm_client'] = SimpleNamespace(model='offline', chat=chat)
    result = run(value)
    assert result['result']['suitability'] == 'needs_information'
    assert 'No selected source images' in json.dumps(received[1])
    assert len(image_parts(received[1])) == 3


def test_partial_front_coverage_marks_missing_tiles_instead_of_repeating_frames(temporal_scene):
    from tools.tool04_edit_planner.observations import sample_indices
    indices = sample_indices(24, 12)
    root = Path(temporal_scene['sample_root']) / 'original_images/CAM_F0'
    for index in indices[1:]:
        (root / f'{index:02d}.png').unlink()
    value = arguments(temporal_scene)
    value['_llm_client'] = client(dict(suitability='needs_information',
                                       reasons=['Need more source frames.'],
                                       checks=dict(road='unknown', lanes='unknown', space='unknown'),
                                       adapted_plan=None, retrieval_suggestion=None))
    result = run(value)
    sheets = result['result']['evidence']['visual_observation']['sheets']
    assert len(sheets) == 3 and sum(sheet['available_frames'] for sheet in sheets) == 1
    assert all(len(sheet['frame_indices']) == 4 for sheet in sheets)


def test_short_scene_uses_only_actual_frames_without_padding_copies(scene):
    value = arguments(scene)
    value['_llm_client'] = client(review())
    result = run(value)
    sheets = result['result']['evidence']['visual_observation']['sheets']
    assert len(sheets) == 1
    assert sheets[0]['frame_indices'] == [0] and sheets[0]['available_frames'] == 1


# --- data-first structural pre-check (structure.py) ---

def frame(timestamp, x, y):
    return dict(timestamp=timestamp, ego2global=[[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]],
                anns=dict(track_tokens=['vehicle-A'], gt_names=['vehicle'],
                          gt_boxes=[[0, 10, 0, 4, 2, 1.5, 0]], gt_velocity_3d=[[0, 2, 0]]),
                cams={'CAM_F0': {'data_path': ''}})


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
