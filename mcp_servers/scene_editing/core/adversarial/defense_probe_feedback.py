"""Bounded defense-constructor evidence for attack planning, never a feasibility gate."""
from pathlib import Path
import json
import shutil
import time

ROOT = Path(__file__).resolve().parent
VERSION = 'defense_probe_feedback.v1'


def read(path):
    return json.loads(Path(path).read_text())


def constructor_binding(scene, target_track, acceptance, count):
    """Bind full scene/map semantics and policy without hashing source data."""
    scene = Path(scene)
    metadata = {}
    for name in ('frames.json', 'map_features.json', 'map_reference.json',
                 'map_drivable_official.json', 'no_cross_lines.json'):
        path = scene/'metadata'/name
        metadata[name] = read(path) if path.is_file() else None
    implementation = []
    for name in ('run_ego_reference.py', 'ego_action_tool.py', 'ego_trajectory_edit.py',
                 'ego_road_checks.py', 'acceptance_policy.py', 'defense_probe_feedback.py',
                 'vehicle_initial_state.py', 'vehicle_motion.py', 'vehicle_control_search.py',
                 'defense_behavior.py', 'defense_behavior_preferences.json'):
        path = ROOT/name
        stat = path.stat()
        implementation.append(dict(path=str(path), size=stat.st_size, modified_ns=stat.st_mtime_ns))
    return dict(version=VERSION, scene_metadata=metadata, target_track=target_track,
                acceptance_policy=acceptance, max_candidates=count, implementation=implementation)



def witness_metrics(row):
    """Report both native samples and delivered-curve extrema without understating either."""
    native = row['max_acceleration_mps2']
    generated = (row.get('generation_curve') or {}).get('max_acceleration_mps2')
    return dict(candidate_id=row['candidate_id'],
        defense_candidate_id='constructor_'+row['candidate_id'].removeprefix('candidate_'),
        min_target_clearance_m=row['min_target_clearance_m'],
        native_max_acceleration_mps2=native,
        generated_curve_max_acceleration_mps2=generated,
        max_acceleration_mps2=max(native, generated) if generated is not None else native)


def summarize(summary, constructor_run):
    if summary.get('status') != 'reference_screen_complete':
        raise ValueError('Defense constructor did not complete')
    rows = summary['candidates']
    passing = [row for row in rows if row['reference_screen_passed']]
    if summary['qualified_count'] != len(passing):
        raise ValueError('Defense constructor passing count contradicts candidate records')
    failures = {}
    for row in rows:
        for check, passed in row['shared_checks'].items():
            if not passed:
                failures[check] = failures.get(check, 0)+1
    diagnostic = summary['defense_difficulty']
    return dict(status='completed', constructor_run=str(constructor_run),
        outcome='reference_witness_found' if passing else 'no_reference_witness_found',
        evaluated_candidates=len(rows), qualified_count=len(passing),
        construction_rejected_count=len(summary.get('construction_failures', [])),
        verified_passing_candidate_ids=[row['candidate_id'] for row in passing],
        witnesses=[witness_metrics(row) for row in passing[:3]],
        failed_check_counts=failures,
        edit_start_clearance_m=diagnostic.get('edit_start_clearance_m'),
        straight_line_stop_estimate=diagnostic.get('straight_line_stop_estimate'),
        diagnostic_only=True, attack_acceptance_affected=False, global_feasibility_proven=False,
        scope='Bounded existing constructor with the unchanged reference policy and complete recorded observations. '
              'A passing witness establishes reference eligibility only; no witness is not a global infeasibility proof.')


def collect(scene, target_track, acceptance, output, count):
    # Imports are local: this helper is also used by lightweight host adapters.
    import acceptance_policy
    from experiment_runtime import atomic_json
    from run_adversarial_loop import run_constructor
    effective = acceptance_policy.resolve(acceptance)
    output = Path(output)
    started = time.perf_counter()
    run_constructor(scene, target_track, output, effective, count, 300)
    result = summarize(read(output/'reference_screen/summary.json'), output)
    result['elapsed_s'] = round(time.perf_counter()-started, 3)
    atomic_json(output/'probe-binding.json', constructor_binding(scene, target_track, effective, count))
    atomic_json(output/'probe-feedback.json', result)
    return result


def reuse(probe_result_path, scene, target_track, acceptance, count, output):
    """Copy a compatible pool into the defense task; the editor rechecks every path."""
    if probe_result_path is None:
        return False
    probe = read(probe_result_path)
    feedback = probe.get('defense_feasibility') or {}
    if feedback.get('status') != 'completed':
        return False
    source = Path(feedback['constructor_run'])
    expected = constructor_binding(scene, target_track, acceptance, count)
    if read(source/'probe-binding.json') != expected:
        return False
    summary = read(source/'reference_screen/summary.json')
    verified = summarize(summary, source)
    if verified['qualified_count'] != feedback['qualified_count']:
        raise ValueError('Defense probe feedback differs from its candidate pool')
    output = Path(output)
    shutil.copytree(source, output)
    summary['pool_provenance'] = dict(source='attack_probe_constructor',
        source_constructor_run=str(source), uses_complete_future_observations=True,
        contains_validated_witness=bool(verified['qualified_count']),
        agent_independent_rediscovery_established=False)
    (output/'reference_screen/summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'reuse-provenance.json').write_text(json.dumps(dict(
        status='reused', source_probe=str(probe_result_path), source_constructor_run=str(source),
        scope='Same scene/map/policy/runtime semantics; defense editor independently rechecks every candidate'), indent=2)+'\n')
    return True
