"""Tool04 data-only structural review and ranked-scene fallback tests."""
from copy import deepcopy
import base64
import json
from pathlib import Path

import pytest

from tools.registry import call_tool
from tools.tool04_edit_planner.real import run
from mcp_servers.scene_editing.adapter.schemas import AttackInput

PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jMZkAAAAASUVORK5CYII=')
ATTACK = dict(map_context='t_junction', ego_action='turn_right', ego_lane_keeping='keep',
              other_origin='right_arm', other_action='turn_left', other_lane_keeping='keep')
ATTACK_NONE = dict(map_context='curve', ego_action='straight', ego_lane_keeping='keep',
                   other_origin='none', other_action='none', other_lane_keeping='none')


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    """Tool04 must not construct a model client after visual review is removed."""
    def blocked(*args, **kwargs):
        raise AssertionError('tool04 is data-only and must not construct LLMClient')
    monkeypatch.setattr('agent.llm_client.LLMClient', blocked)


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
                       retrieval_query=f'白天路口，自车直行接近，右侧支路清晰 {index}',
                       attack_edit_strategy=None if no_other else '在右侧支路添加一辆目标车辆',
                       ego_input_edit_strategy='调整自车接近速度与到达时刻',
                       ego_correct_trajectory_strategy='自车减速后安全通过')
                  for index in (1, 2, 3)]
    return ScenarioOutput.model_validate(dict(
        target_scene=('多云白天，弯道，自车直行，无其他车辆。' if no_other
                      else '多云白天，丁字路口，自车直行接近，右侧支路一辆车直行。'),
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


def stub_map(monkeypatch, checks, facts=None, available=True, reasons=('measured map verdict',),
             suggestion=None):
    from tools.tool04_edit_planner import mapfacts
    monkeypatch.setattr(mapfacts, 'collect', lambda frames, attack: dict(
        available=available, checks=dict(checks), facts=facts if facts is not None else {'stub': True},
        reasons=list(reasons), evidence_limits=[], retrieval_suggestion=suggestion))


def test_ready_copies_target_and_attack_and_builds_plan_without_a_model(scene, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
    result = call_tool('edit_planner', arguments(scene), mode='real')
    assert result['status'] == 'success', result
    output = result['result']
    assert set(output) == set(__import__('tools.tool04_edit_planner.schemas', fromlist=['Output']).Output.model_fields)
    assert output['target_scene'] == scenario()['target_scene']
    assert output['attack'] == ATTACK
    assert output['candidate_id'] == 'cand_1'
    assert (output['scene_id'], output['scene_token']) == ('source', 'token')
    assert output['suitability'] == 'ready'
    assert output['checks'] == {'road': 'pass', 'lanes': 'pass', 'space': 'pass'}
    assert output['adapted_plan']['ego_correct_trajectory_strategy'] == '自车减速后安全通过'
    assert output['adapted_plan']['ego_input_edit_strategy']
    handoff = AttackInput.model_validate(output['attack_input'])
    assert handoff.operation in ('edit_existing', 'auto')
    assert (handoff.scene_ref.scene_id, handoff.scene_ref.scene_token) == ('source', 'token')
    assert output['evidence']['visual_model_used'] is False
    assert output['evidence']['verification_source'] == 'measured_map_precheck'


def test_plan_uses_the_measured_interaction_path(scene, monkeypatch):
    facts = dict(stub=True, interaction_path_pairs=[dict(
        paths_cross=True, interaction_compatible=True, on_official_network=True,
        ego_path_kind='approach_straight:18121', other_connector_fid=20045)])
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'}, facts=facts)
    result = run(arguments(scene))
    assert result['status'] == 'success', result
    plan = result['result']['adapted_plan']
    assert '车道 18121' in plan['ego_input_edit_strategy']
    assert 'connector 20045' in plan['ego_input_edit_strategy']


def test_map_failure_rejects_without_a_model(scene, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'fail', 'space': 'pass'},
             suggestion='检索具有所需支路连接的场景。')
    result = run(arguments(scene))
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'unsuitable'
    assert output['adapted_plan'] is None
    assert output['retrieval_suggestion'] == '检索具有所需支路连接的场景。'
    assert output['evidence']['visual_model_used'] is False


def test_incomplete_map_returns_needs_information_without_visual_fallback(scene, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'unknown', 'space': 'unknown'})
    result = run(arguments(scene))
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'needs_information'
    assert output['adapted_plan'] is None
    assert output['checks'] == {'road': 'pass', 'lanes': 'unknown', 'space': 'unknown'}
    assert output['evidence']['visual_model_used'] is False


def test_missing_images_do_not_block_the_data_only_review(scene, monkeypatch):
    Path(scene['sample_root'], 'frame.png').unlink()
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
    result = run(arguments(scene))
    assert result['status'] == 'success', result
    assert result['result']['suitability'] == 'ready'
    assert result['result']['evidence']['source_images'] == []


def test_missing_render_asset_does_not_block_the_structural_review(scene, monkeypatch):
    scene['asset_path'] = str(Path(scene['sample_root']) / 'no-asset')
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
    result = run(arguments(scene))
    assert result['result']['suitability'] == 'ready'
    assert result['result']['evidence']['gaussian_assets'] == []


def test_no_target_vehicle_keeps_attack_edit_null(scene, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
    result = run(arguments(scene, attack=ATTACK_NONE))
    assert result['status'] == 'success', result
    assert result['result']['attack'] == ATTACK_NONE
    assert result['result']['adapted_plan']['attack_edit_strategy'] is None
    assert result['result']['attack_input'] is None


@pytest.mark.parametrize('form', ['envelope', 'result', 'result_ref'])
def test_scenario_and_retrieval_accept_three_forms(scene, tmp_path, form, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
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
    result = call_tool('edit_planner', value, mode='real')
    assert result['status'] == 'success', result
    output = result['result']
    assert (output['scene_id'], output['scene_token']) == ('source', 'token')
    assert output['evidence']['retrieval']['candidate_id'] == 'cand_1'


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


def test_supplied_vehicle_asset_reaches_tool05_handoff(scene, tmp_path, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
    asset = tmp_path / 'vehicle.ply'
    asset.write_bytes(b'ply fixture')
    value = arguments(scene)
    value['vehicle'] = dict(asset_path=str(asset), size_m=[4.4, 2.0, 1.6])
    result = run(value)
    assert result['status'] == 'success', result
    handoff = AttackInput.model_validate(result['result']['attack_input'])
    assert handoff.vehicle is not None and handoff.vehicle.asset_path == str(asset)
    assert handoff.operation in ('edit_existing', 'auto')


def test_legacy_input_fields_are_rejected(scene):
    for extra in ('scene_requirement', 'target_effect', 'edit_strategy', 'interaction',
                  'retrieval_result_ref'):
        value = arguments(scene)
        value[extra] = 'legacy'
        assert run(value)['error']['type'] == 'input_validation_error', extra


def test_mock_claims_no_inspection(scene):
    result = call_tool('edit_planner', arguments(scene), mode='mock')
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'needs_information'
    assert output['checks'] == {'road': 'unknown', 'lanes': 'unknown', 'space': 'unknown'}
    assert output['adapted_plan'] is None


def test_evidence_keeps_the_review_inputs(scene, monkeypatch):
    stub_map(monkeypatch, {'road': 'pass', 'lanes': 'pass', 'space': 'pass'})
    result = run(arguments(scene))
    evidence = result['result']['evidence']
    assert evidence['eligible_vehicle_tracks'] == ['vehicle-A']
    assert evidence['retrieval']['candidate_id'] == 'cand_1'
    assert evidence['frame_indices'] == [0]


def test_rank_one_failure_falls_back_to_rank_two(scene, monkeypatch):
    from tools.tool04_edit_planner import mapfacts
    answers = iter([
        dict(available=True, checks={'road': 'fail', 'lanes': 'pass', 'space': 'pass'},
             facts={'rank': 1}, reasons=['first scene lacks the right arm'],
             evidence_limits=[], retrieval_suggestion='use another ranked scene'),
        dict(available=True, checks={'road': 'pass', 'lanes': 'pass', 'space': 'pass'},
             facts={'rank': 2}, reasons=['second scene supports the target'],
             evidence_limits=[], retrieval_suggestion=None),
    ])
    monkeypatch.setattr(mapfacts, 'collect', lambda frames, attack: next(answers))
    value = arguments(scene)
    fallback = deepcopy(scene)
    fallback.update(scene_id='source-2', scene_token='token-2')
    value['retrieval_result']['alternatives'] = [dict(
        rank=2, scene_ref=fallback, video_path='/retrieved/second.mp4',
        source_file='/indexed/second.mp4', match_score=.7,
        source_segment={'start': 1., 'end': 3.})]
    result = run(value)
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'ready'
    assert output['scene_token'] == 'token-2'
    assert output['attack_input']['scene_ref']['scene_token'] == 'token-2'
    assert output['evidence']['selected_rank'] == 2
    assert [item['rank'] for item in output['evidence']['retrieval_attempts']] == [1, 2]


def test_all_unsuitable_returns_rank_one_verdict_with_attempt_history(scene, monkeypatch):
    from tools.tool04_edit_planner import mapfacts
    monkeypatch.setattr(mapfacts, 'collect', lambda frames, attack: dict(
        available=True, checks={'road': 'pass', 'lanes': 'fail', 'space': 'pass'},
        facts={}, reasons=['missing right arm'], evidence_limits=[],
        retrieval_suggestion='retrieve a clearer right arm'))
    value = arguments(scene)
    fallback = deepcopy(scene)
    fallback.update(scene_id='source-2', scene_token='token-2')
    value['retrieval_result']['alternatives'] = [dict(
        rank=2, scene_ref=fallback, video_path='/retrieved/second.mp4',
        source_file='/indexed/second.mp4', match_score=.7,
        source_segment={'start': 1., 'end': 3.})]
    result = run(value)
    assert result['status'] == 'success', result
    output = result['result']
    assert output['suitability'] == 'unsuitable'
    assert output['scene_token'] == 'token'
    assert output['evidence']['selected_rank'] is None
    assert [item['rank'] for item in output['evidence']['retrieval_attempts']] == [1, 2]


def test_raw_map_reader_returns_measured_facts_for_a_real_sample():
    from tools.tool04_edit_planner import mapfacts
    if mapfacts.python_path() is None:
        pytest.skip('simulator interpreter is not available')
    frames_path = Path('data/navtrain_ep_per1/samples/5863c4eb96825e85/metadata/frames.json')
    if not frames_path.is_file():
        pytest.skip('sample scene is not available')
    result = mapfacts.collect(json.loads(frames_path.read_text()),
                              dict(map_context='t_junction', ego_action='turn_right', ego_lane_keeping='keep',
                                   other_origin='right_arm', other_action='turn_left', other_lane_keeping='keep'))
    assert result['available'] is True, result
    facts = result['facts']
    assert set(result['checks']) == {'road', 'lanes', 'space'}
    assert set(result['checks'].values()) <= {'pass', 'fail', 'unknown'}
    assert facts['approach_lane_count'] > 0
    assert facts['connector_count'] > 0
    assert 'lane_count_by_group' in facts and 'target_junction' in facts
    assert facts['selection_policy'] == 'first_real_junction_on_recorded_route'
