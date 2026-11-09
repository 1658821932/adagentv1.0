"""Semantic PDM result binding; raw scene data is compared directly, never hashed."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
RUNNER_VERSION = 'pdm_closed_loop.probe.v4'


def binding(arguments, settings):
    scene = Path(arguments['challenge_baseline_scene_path']).resolve()
    frames = json.loads((scene/'metadata/frames.json').read_text())
    location = frames[0]['map_location']
    map_root = Path(settings['map_root']).resolve()
    metadata = json.loads((map_root/'nuplan-maps-v1.0.json').read_text())[location]
    map_files = []
    for path in sorted((map_root/location).rglob('*.gpkg')):
        stat = path.stat()
        map_files.append(dict(path=str(path), size=stat.st_size, modified_ns=stat.st_mtime_ns))
    implementation = []
    for path in [ROOT/'run_pdm_closed_loop.py', ROOT/'pdm_binding.py', ROOT/'pdm_traffic_lights.py', ROOT/'pdm_temporal.py']:
        stat = path.stat()
        implementation.append(dict(path=str(path),size=stat.st_size,modified_ns=stat.st_mtime_ns))
    baseline = dict(runner_version=RUNNER_VERSION, implementation=implementation,
        source_scene=str(scene), frames=frames, runtime=dict(settings),
        history_s=arguments['challenge_history_s'],target_track=arguments['challenge_target_track'],
        ignore_traffic_lights=arguments.get('challenge_ignore_traffic_lights',False),
        target_is_new=arguments['challenge_target_is_new'], map_metadata=metadata,map_files=map_files)
    attack = json.loads((Path(arguments['challenge_attack_scene_path'])/'metadata/frames.json').read_text())
    return dict(baseline=baseline,attack_frames=attack,
        run_baseline_diagnostic=arguments.get('challenge_run_baseline_diagnostic',False))


def matching_baseline(cache, current):
    cache = Path(cache)
    try:
        previous=json.loads((cache/'binding.json').read_text())
        summary=json.loads((cache/'baseline/summary.json').read_text())
        return (previous['baseline']==current['baseline'] and summary['status']=='completed'
            and summary['contact_samples']==0 and summary['road_violation_samples']==0)
    except (OSError, ValueError, KeyError, TypeError):
        return False
