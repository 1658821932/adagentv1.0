"""Validated native visual lane-change configuration; no metric claims."""
from __future__ import annotations

import json
import math
from pathlib import Path


def validate(config: dict) -> dict:
    if config.get('backend') != 'ufo' or config.get('operation') != 'lane_change':
        raise ValueError('UFO config must specify backend=ufo and operation=lane_change')
    if config.get('direction') not in {'left', 'right'}:
        raise ValueError('UFO lane change requires explicit direction=left or right')
    for key, default in [('start_fraction', .1), ('end_fraction', .9), ('vehicle_widths', 1.2)]:
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        config[key] = value
    if not 0 <= config['start_fraction'] < config['end_fraction'] <= 1:
        raise ValueError('Require 0 <= start_fraction < end_fraction <= 1')
    if not 0 < config['vehicle_widths'] <= 2:
        raise ValueError('vehicle_widths must be in (0, 2] to limit disocclusion artifacts')
    if not config.get('target_selection_path'):
        raise ValueError('UFO lane configuration requires target_selection_path')
    return config


def generate(input_data: dict) -> dict:
    from tools.tool05_instance_discovery.selection import load_bound_instances
    if input_data.get('operation', 'lane_change') != 'lane_change' or input_data.get('output_kind', 'rigid_config') != 'rigid_config':
        raise ValueError('UFO supports only one existing-vehicle visual lane_change')
    if input_data.get('frame_count', 40) != 40:
        raise ValueError('UFO Agent editing requires frame_count=40')
    if input_data.get('asset_path') or input_data.get('asset_scale'):
        raise ValueError('UFO visual editing does not add or replace assets')
    instance_path = Path(input_data['instance_info_path']).expanduser().resolve()
    instances = json.loads(instance_path.read_text())
    binding_path = instances.get('target_selection_path')
    if not binding_path:
        raise ValueError('Run target_selector and use selected_instance_info_path')
    binding, selected = load_bound_instances(binding_path)
    if Path(binding['selected_instance_info_path']).resolve() != instance_path or instances.get('instances') != [selected]:
        raise ValueError('Selected instance file differs from binding')
    config = validate({'backend': 'ufo', 'operation': 'lane_change',
                       'direction': input_data.get('direction'),
                       'start_fraction': input_data.get('start_fraction', .1),
                       'end_fraction': input_data.get('end_fraction', .9),
                       'vehicle_widths': input_data.get('vehicle_widths', 1.2),
                       'target_selection_path': str(Path(binding_path).resolve()),
                       'selected_instance_id': selected['instance_id'],
                       'edit_plan': input_data.get('edit_plan'),
                       'metric_calibrated': False, 'profile': 'view_consistent_v2'})
    output = Path(input_data['output_path']).expanduser().resolve()
    if output.suffix.lower() != '.json' or output in {instance_path, Path(binding_path).resolve(), Path(binding['candidate_info_path']).resolve()}:
        raise ValueError('Use a distinct .json output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    return {'saved_path': str(output), 'edit_config_path': str(output), **config}
