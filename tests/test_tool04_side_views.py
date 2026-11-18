"""Side-view review uses the frames nearest the chosen junction, on the real sample."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.registry import call_tool
from tools.tool02_scene_instance_generator.schemas import Output as ScenarioOutput

SAMPLE = Path('data/navtrain_ep_per1/samples/5863c4eb96825e85')


def scene():
    if not (SAMPLE / 'metadata/frames.json').is_file():
        pytest.skip('sample scene is not available')
    frames = json.loads((SAMPLE / 'metadata/frames.json').read_text())
    return dict(scene_id='5863c4eb96825e85', scene_token='5863c4eb96825e85', dataset='navtrain',
                sample_root=str(SAMPLE), matched_camera='CAM_F0', frame_count=len(frames), fps=2.0,
                manifest_path=str(SAMPLE / 'manifest.json'),
                frames_path=str(SAMPLE / 'metadata/frames.json'),
                cameras_path=str(SAMPLE / 'metadata/cameras.json'), cameras={},
                asset_path=str(SAMPLE / 'asset'), images_complete=False)


def scenario():
    return ScenarioOutput.model_validate(dict(
        target_scene='多云白天，十字路口，自车右转，本次场景没有其他车辆。',
            attack=dict(map_context='cross_intersection', ego_action='turn_right', ego_lane_keeping='keep',
                        other_origin='none', other_action='none', other_lane_keeping='none'),
        candidates=[dict(candidate_id=f'cand_{index}', base_video_requirement=f'路口条件 {index}',
                         attack_edit_strategy=None, ego_input_edit_strategy=None,
                         ego_correct_trajectory_strategy='自车减速通过') for index in (1, 2, 3)])
                ).model_dump(mode='json')


class Client:
    model = 'offline'

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        value = self.answers.pop(0)
        return json.dumps(value)


def arguments():
    return dict(scenario_result=scenario(), candidate_id='cand_1',
                retrieval_result=dict(scene_ref=scene(), mode='real', video_path=None,
                                      candidate_id='cand_1'))


def test_side_view_uses_the_frames_near_the_chosen_junction():
    client = Client([dict(action='request_views', cameras=['CAM_L0'], purpose='核对目标路口空间'),
                     dict(suitability='needs_information', reasons=['目标路口方向证据不足。'],
                          checks=dict(road='unknown', lanes='unknown', space='unknown'),
                          adapted_plan=None, retrieval_suggestion=None)])
    value = arguments()
    value['_llm_client'] = client
    result = call_tool('edit_planner', value, mode='real')
    assert result['status'] == 'success', result
    output = result['result']
    assert (output['target_scene'], output['attack']) == (
        arguments()['scenario_result']['target_scene'], arguments()['scenario_result']['attack'])
    evidence = output['evidence']
    assert evidence['map_available'] is True, 'the raw map must be readable for this sample'
    target = evidence.get('target_region_frames')
    assert target, 'the reviewed junction must yield frames near the target region'
    sheets = evidence['visual_observation']['sheets']
    side = [sheet for sheet in sheets if sheet['camera'] == 'CAM_L0']
    assert side, 'side camera must be delivered'
    assert side[0]['frame_indices'] == sorted(target), (side[0]['frame_indices'], sorted(target))
    assert side[0]['frame_indices'] != evidence['frame_indices'], 'the side view must not reuse the timeline sample'
    assert client.calls == 2


def test_side_view_falls_back_when_no_target_region_is_reliable(monkeypatch):
    """Separate case: without a reliable target junction the review keeps the normal sampling."""
    from tools.tool04_edit_planner import mapfacts

    monkeypatch.setattr(mapfacts, 'collect', lambda frames, attack: dict(
        available=False, checks=dict(road='unknown', lanes='unknown', space='unknown'),
        facts=None, reasons=['raw map unavailable'], evidence_limits=['raw map unavailable'],
        retrieval_suggestion=None))
    client = Client([dict(action='request_views', cameras=['CAM_L0'], purpose='核对侧向空间'),
                     dict(suitability='needs_information', reasons=['缺少目标区域证据。'],
                          checks=dict(road='unknown', lanes='unknown', space='unknown'),
                          adapted_plan=None, retrieval_suggestion=None)])
    value = arguments()
    value['_llm_client'] = client
    result = call_tool('edit_planner', value, mode='real')
    assert result['status'] == 'success', result
    evidence = result['result']['evidence']
    assert evidence.get('target_region_frames') in (None, [])
    side = [sheet for sheet in evidence['visual_observation']['sheets'] if sheet['camera'] == 'CAM_L0']
    delivered = sorted({index for sheet in side for index in sheet['frame_indices']})
    assert side and delivered == sorted(evidence['frame_indices'])
    assert client.calls == 2
