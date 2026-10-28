"""Regression coverage for migrating UFO into main's typed planning interface."""
import json
from pathlib import Path
from unittest.mock import patch

from agent.executor import ToolExecutor
from tools.tool07_edit_plan_builder.real import run


class Client:
    model = 'offline-fixture'

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return json.dumps(self.result)


def insert_plan(asset):
    return {'edit_plan': 'Insert the supplied car at the requested position.',
            'target_instances': [], 'operations': [
                {'operation': 'insert', 'object_id': 'new_car',
                 'asset_manifest_path': str(asset), 'position_m': [20, 3, 0],
                 'yaw_degrees': 0},
                {'operation': 'translate', 'target_id': 'new_car',
                 'offset_m': [0, -3, 0], 'start_seconds': 0, 'end_seconds': 1.8}],
            'constraints': ['Estimated metric units; inspect rendered grounding.']}


def test_metric_plan_retains_main_typed_validation_and_compiles(tmp_path):
    asset, calibration = tmp_path/'asset.json', tmp_path/'calibration.json'
    asset.write_text('{}')
    calibration.write_text('{}')
    client = Client(insert_plan(asset))
    result = run({'backend': 'ufo', 'operation': 'metric_edit',
                  'edit_request': 'Insert this car 20m ahead and 3m left, then move right 3m from 0 to 1.8s.',
                  'calibration_path': str(calibration), 'asset_manifest_path': str(asset),
                  '_llm_client': client})
    assert result['status'] == 'success', result
    assert result['meta']['attempts'] == 1
    executor = ToolExecutor({'edit_config_generator': 'real'}, reconstruction_backend='ufo')
    config = executor.execute('edit_config_generator', {
        'operation': 'metric_edit', 'calibration_path': str(calibration),
        'operations': result['result']['operations'], 'output_path': str(tmp_path/'edit.json')})
    assert config['status'] == 'success', config
    saved = json.loads((tmp_path/'edit.json').read_text())
    assert saved['operations'] == result['result']['operations']
    assert saved['operations'][1]['offset_m'] == [0, -3, 0]


def test_metric_plan_cannot_substitute_unsupplied_asset(tmp_path):
    asset = tmp_path/'asset.json'
    response = insert_plan(tmp_path/'invented.json')
    client = Client(response)
    result = run({'backend': 'ufo', 'operation': 'metric_edit', 'edit_request': 'Insert this car.',
                  'calibration_path': '/scene/calibration.json', 'asset_manifest_path': str(asset),
                  '_llm_client': client})
    assert result['status'] == 'error'
    assert result['error']['type'] == 'output_validation_error'
    assert client.calls == 2


def test_unbound_existing_actor_is_rejected_before_config_or_worker(tmp_path):
    plan = insert_plan(tmp_path/'asset.json')
    plan['operations'] = [{**plan['operations'][1], 'target_id': 'selected'}]
    client = Client(plan)
    result = run({'backend': 'ufo', 'operation': 'metric_edit', 'edit_request': 'Move the front car right 3m.',
                  'calibration_path': '/scene/calibration.json', '_llm_client': client})
    assert result['status'] == 'error'
    assert 'target_selection_path' in json.dumps(result)


def test_dggt_planner_still_uses_main_typed_contract():
    client = Client({'edit_plan': 'Move the supplied car.', 'target_instances': [7],
                     'operations': [{'operation': 'modify_trajectory', 'target': 7, 'parameters': {}}],
                     'constraints': []})
    result = run({'backend': 'dggt', 'edit_request': 'Move car 7.',
                  'instances': [{'instance_id': 7}], '_llm_client': client})
    assert result['status'] == 'success', result
    assert result['result']['target_instances'] == [7]


def test_failed_reconstruction_drops_previous_asset():
    executor = ToolExecutor({'reconstruction': 'real'})
    executor.reconstruction = {'workspace': '/old-scene'}
    with patch('agent.executor.call_tool', return_value={
        'status': 'error', 'tool': 'reconstruction', 'error': {'message': 'failed'}}):
        executor.execute('reconstruction', {'video_path': '/new.mp4'})
    assert executor.reconstruction is None
