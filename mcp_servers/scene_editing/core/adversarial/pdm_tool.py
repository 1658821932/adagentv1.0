"""Independent PDM baseline/attack challenge with no AdAgent runtime dependency."""
import json
import os
import subprocess
import shutil
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((BACKEND_ROOT/'backend_settings.json').read_text())


def run_pdm_challenge(arguments, output_dir, log_path):
    settings = SETTINGS['pdm']
    output = Path(output_dir)
    if arguments.get('challenge_probe_result_path'):
        return reuse_probe(arguments,output,log_path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite PDM output")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=settings['worldengine'])
    env['LD_LIBRARY_PATH'] = settings['system_lib']+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    threads = str(settings.get('cpu_threads',1))
    env.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)

    command = [settings['python'], str(BACKEND_ROOT/'run_pdm_closed_loop.py'),
        '--baseline', arguments['challenge_baseline_scene_path'], '--attack', arguments['challenge_attack_scene_path'],
        '--output', str(output), '--map-root', settings['map_root'],
        '--history-s', str(arguments['challenge_history_s']), '--target-track', arguments['challenge_target_track']]
    if arguments.get('challenge_ignore_traffic_lights',False):
        command.append('--ignore-traffic-lights')
    if arguments['challenge_target_is_new']:
        command.append('--target-is-new')
    if arguments.get('challenge_run_baseline_diagnostic', False):
        command.append('--baseline-diagnostic')
    if arguments.get('challenge_run_baseline_diagnostic', False) and arguments.get('challenge_baseline_cache_path'):
        command.extend(['--baseline-cache',arguments['challenge_baseline_cache_path']])
    with Path(log_path).open('x') as log:
        completed = subprocess.run(command, cwd=BACKEND_ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, timeout=settings.get('timeout_seconds', 300))
    if completed.returncode:
        raise RuntimeError('PDM subprocess failed; see pdm.log')
    return read_result(output)


def read_result(output):
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    attack = summary['results']['attack']
    observations = json.loads((output/'attack/observations.json').read_text())
    if not observations or not observations[0].get('target_in_pdm'):
        raise RuntimeError('Attack target did not enter the initial PDM observation')
    if any(row['observation_timestamp_us'] > row['max_history_time_us'] for row in observations):
        raise RuntimeError('PDM received a future observation')
    safe = not attack['contact_samples'] and not attack['road_violation_samples']
    return dict(summary=summary, pdm_defense_succeeded=safe)


def reuse_probe(arguments, output, log_path):
    """Reuse only a completed probe of this exact scene and runtime configuration."""
    try:
        from .pdm_binding import binding
    except ImportError:
        from pdm_binding import binding
    started = time.perf_counter()
    probe_path = Path(arguments['challenge_probe_result_path']).resolve()
    probe = json.loads(probe_path.read_text())
    source = Path(probe['pdm_output']).resolve()
    expected = binding(arguments,SETTINGS['pdm'])
    actual = json.loads((source/'binding.json').read_text())
    if probe.get('status') != 'completed' or actual != expected:
        raise ValueError('Candidate PDM probe differs from submitted scene/history/target/runtime/map')
    result = read_result(source)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for name in result['summary']['results']:
        shutil.copytree(source/name,output/name)
    for name in ('summary.json','binding.json','config.json'):
        shutil.copyfile(source/name,output/name)
    summary = dict(result['summary'],probe_reused=True,probe_result_path=str(probe_path),
                   reuse_wall_seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    Path(log_path).write_text('Reused verified candidate PDM probe: '+str(probe_path)+'\n')
    return dict(summary=summary,pdm_defense_succeeded=result['pdm_defense_succeeded'])
