"""Tool04 no longer calls a visual model or generates side-view contact sheets."""
import json
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
        target_scene='多云白天，十字路口，自车直行接近，本次场景没有其他车辆。',
        attack=dict(map_context='cross_intersection', ego_action='straight', ego_lane_keeping='keep',
                    other_origin='none', other_action='none', other_lane_keeping='none'),
        candidates=[dict(candidate_id=f'cand_{index}', base_video_requirement=f'路口条件 {index}',
                         retrieval_query=f'白天路口，自车直行接近，右侧支路清晰 {index}',
                         attack_edit_strategy=None, ego_input_edit_strategy=None,
                         ego_correct_trajectory_strategy='自车减速通过') for index in (1, 2, 3)])
    ).model_dump(mode='json')


def arguments():
    return dict(scenario_result=scenario(), candidate_id='cand_1',
                retrieval_result=dict(scene_ref=scene(), mode='real', video_path=None,
                                      candidate_id='cand_1'))


def test_review_does_not_build_visual_observation_or_call_a_model():
    result = call_tool('edit_planner', arguments(), mode='real')
    assert result['status'] == 'success', result
    output = result['result']
    assert result['meta']['model'] is None
    assert result['meta']['attempts'] == 0
    assert output['evidence']['visual_model_used'] is False
    assert 'visual_observation' not in output['evidence']
    assert 'visual_review' not in output['evidence']


def test_target_region_frames_remain_available_as_map_evidence():
    result = call_tool('edit_planner', arguments(), mode='real')
    assert result['status'] == 'success', result
    assert result['result']['evidence'].get('map_available') is True
    assert result['result']['evidence'].get('target_region_frames') is not None
