"""Explicit fresh editor run with compatible completed understanding/constructor."""
from pathlib import Path
import json
import shutil


def reuse_understanding(source_run, arguments, acceptance, count, destination):
    from defense_probe_feedback import constructor_binding
    source_run = Path(source_run).resolve()
    payload = json.loads((source_run/'input/stage_request.json').read_text())
    if payload['stage'] != 'ego_editing':
        raise ValueError('Completed-stage source is not an ego editing run')
    previous = payload['arguments']
    for key in ('attack_result', 'target_effect', 'allow_safe_stop', 'use_foci'):
        if previous[key] != arguments[key]:
            raise ValueError('Completed-stage source changed task/attack: '+key)
    attack = arguments['attack_result']
    expected = constructor_binding(attack['frozen_scene_path'], attack['target_track'], acceptance, count)
    actual = json.loads((source_run/'output/defense_constructor/probe-binding.json').read_text())
    if actual != expected:
        raise ValueError('Completed stages differ from scene/map/policy/runtime')
    source = source_run/'output/defense_understanding'
    summary = json.loads((source/'summary.json').read_text())
    if summary.get('status') != 'completed':
        raise ValueError('Previous understanding did not complete')
    config = json.loads((source/'config.json').read_text())
    if Path(config['scene_dir']).resolve() != Path(attack['frozen_scene_path']).resolve() or config['target_track'] != attack['target_track']:
        raise ValueError('Previous understanding belongs to another scene/target')
    shutil.copytree(source, destination,
        ignore=shutil.ignore_patterns('requests', 'timing', 'status-history', 'tool-exchanges'))
    (Path(destination)/'reuse-provenance.json').write_text(json.dumps(dict(status='reused',
        source_run=str(source_run), source_understanding=str(source),new_model_request=False),indent=2)+'\n')
