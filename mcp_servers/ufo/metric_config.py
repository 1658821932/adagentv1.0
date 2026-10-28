"""Dependency-free metric editing contract shared by the Agent and worker."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path


def number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return float(value)


def vector(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f'{name} must contain three finite coordinates')
    return [number(v, name) for v in value]


def resolve_paths(config: dict, base: Path) -> dict:
    config = copy.deepcopy(config)
    def resolve(value):
        path = Path(value).expanduser()
        return str((base/path if not path.is_absolute() else path).resolve())
    for key in ('calibration_path', 'target_selection_path'):
        if config.get(key):
            config[key] = resolve(config[key])
    for op in config.get('operations', []):
        if op.get('asset_manifest_path'):
            op['asset_manifest_path'] = resolve(op['asset_manifest_path'])
    return config


def validate(config: dict, *, agent=False) -> dict:
    config = copy.deepcopy(config)
    if config.get('backend') != 'ufo' or config.get('operation') != 'metric_edit' or config.get('units') != 'meters':
        raise ValueError('Metric config requires backend=ufo, operation=metric_edit, units=meters')
    if config.get('reference_frame', 'road_at_edit_start') != 'road_at_edit_start':
        raise ValueError('Metric edits require the fixed road_at_edit_start frame')
    config['reference_frame'] = 'road_at_edit_start'
    if config.get('road_geometry','native') not in {'native','planar_proxy'}:
        raise ValueError('road_geometry must be native or explicit planar_proxy')
    if not isinstance(config.get('calibration_path'), str) or not config['calibration_path']:
        raise ValueError('Metric editing requires a persisted calibration_path')
    operations = config.get('operations')
    if not isinstance(operations, list) or not operations:
        raise ValueError('Metric editing requires operations')
    inserted, translated = set(), set()
    selected = False
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError('Each metric operation must be an object')
        if op.get('operation') == 'insert':
            ident = op.get('object_id')
            if not isinstance(ident, str) or not ident or ident == 'selected' or ident in inserted:
                raise ValueError('Inserted object_id must be unique and must not be selected')
            if not isinstance(op.get('asset_manifest_path'), str) or not op['asset_manifest_path']:
                raise ValueError('Insertion requires asset_manifest_path with known units and axes')
            op['position_m'] = vector(op.get('position_m'), 'position_m')
            if abs(op['position_m'][2]) > 1e-9:
                raise ValueError('Road vehicle insertion requires position_m up=0')
            op['yaw_degrees'] = number(op.get('yaw_degrees', 0), 'yaw_degrees')
            op['motion_reference'] = op.get('motion_reference', 'world_fixed')
            if op['motion_reference'] not in {'world_fixed','ego_projected','ego_translation'}:
                raise ValueError('Insertion motion_reference must be world_fixed, ego_projected or ego_translation')
            inserted.add(ident)
        elif op.get('operation') == 'translate':
            ident = op.get('target_id')
            if not isinstance(ident, str) or ident not in inserted | {'selected'}:
                raise ValueError('Translate target_id must be selected or an object inserted earlier in this job')
            if ident in translated:
                raise ValueError('One translation trajectory per object is supported')
            translated.add(ident)
            selected |= ident == 'selected'
            op['offset_m'] = vector(op.get('offset_m'), 'offset_m')
            if abs(op['offset_m'][2]) > 1e-9:
                raise ValueError('Road translation requires zero up offset to retain road height')
            op['start_seconds'] = number(op.get('start_seconds'), 'start_seconds')
            op['end_seconds'] = number(op.get('end_seconds'), 'end_seconds')
            if not 0 <= op['start_seconds'] < op['end_seconds']:
                raise ValueError('Require 0 <= start_seconds < end_seconds')
            if op.get('trajectory_mode', 'offset_from_original') != 'offset_from_original':
                raise ValueError('Only offset_from_original trajectories are supported; no implicit lane hold')
            op['trajectory_mode'] = 'offset_from_original'
        else:
            raise ValueError('Metric operations support insert and translate only')
    if selected:
        if not config.get('target_selection_path'):
            ident = config.get('track_id')
            if agent or type(ident) is not int or ident <= 0:
                raise ValueError('Existing-vehicle metric editing requires target_selection_path (or explicit CLI track_id)')
    return config


def generate(input_data: dict) -> dict:
    """Generate deterministic metric configuration; never guess scale or assets."""
    from tools.tool05_instance_discovery.selection import load_bound_instances
    if input_data.get('frame_count', 40) != 40:
        raise ValueError('UFO Agent metric editing requires frame_count=40')
    if input_data.get('units', 'meters') != 'meters':
        raise ValueError('Metric generator requires units=meters')
    config = {k: input_data[k] for k in ('operations', 'calibration_path', 'target_selection_path','road_geometry') if k in input_data}
    config.update(backend='ufo', operation='metric_edit', units='meters', reference_frame='road_at_edit_start')
    if input_data.get('instance_info_path'):
        path = Path(input_data['instance_info_path']).expanduser().resolve()
        instances = json.loads(path.read_text())
        binding_path = instances.get('target_selection_path')
        if not binding_path:
            raise ValueError('Use selected_instance_info_path, not unselected candidates')
        binding, selected = load_bound_instances(binding_path)
        if Path(binding['selected_instance_info_path']).resolve() != path or instances.get('instances') != [selected]:
            raise ValueError('Selected instance file differs from binding')
        if config.get('target_selection_path') and Path(config['target_selection_path']).resolve() != Path(binding_path).resolve():
            raise ValueError('Metric configuration changed the selected target')
        config['target_selection_path'] = str(Path(binding_path).resolve())
    config = validate(config, agent=True)
    # Validate a persisted calibration before dispatch; the worker verifies its
    # reconstruction fingerprint and numerical contents again in ufoex.
    if not Path(config['calibration_path']).expanduser().is_file():
        raise FileNotFoundError('Calibrate the reconstruction first; calibration_path does not exist')
    for op in config['operations']:
        if op['operation'] == 'insert' and not Path(op['asset_manifest_path']).expanduser().is_file():
            raise FileNotFoundError(f"Asset manifest not found: {op['asset_manifest_path']}")
    output = Path(input_data['output_path']).expanduser().resolve()
    protected = {Path(config['calibration_path']).expanduser().resolve()}
    if config.get('target_selection_path'):
        binding, _ = load_bound_instances(config['target_selection_path'])
        protected.update(Path(binding[k]).resolve() for k in ('candidate_info_path', 'selected_instance_info_path'))
        protected.add(Path(config['target_selection_path']).resolve())
    protected.update(Path(op['asset_manifest_path']).expanduser().resolve() for op in config['operations'] if op['operation'] == 'insert')
    if output.suffix.lower() != '.json' or output in protected:
        raise ValueError('Use a distinct .json configuration output path')
    # Resolve references now so CLI and Agent working directories cannot change their meaning.
    config['calibration_path'] = str(Path(config['calibration_path']).expanduser().resolve())
    for op in config['operations']:
        if op['operation'] == 'insert':
            op['asset_manifest_path'] = str(Path(op['asset_manifest_path']).expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return {'saved_path': str(output), 'edit_config_path': str(output), **config}
