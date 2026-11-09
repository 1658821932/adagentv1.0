"""Independent re-check of a submitted ego candidate from an editor/agent run.

Recomputes, from the scene data and the archived trajectory.csv only (no editor state):
2 Hz target/other clearances, motion peaks, strict lane-union road check (1e-8 m),
official nuPlan map DAC with the 0.1 m positive-margin gate, and a densified
linear-interpolation clearance diagnostic. Writes independent-check.json into the run.

Usage:
  python scripts/run_ego_independent_check.py --input <scene dir> --run <run dir> \
      --candidate edit_003
"""
import acceptance_policy as policy
from workflow_timing import timed_operation

import argparse
import json
from pathlib import Path

import numpy as np

from ego_road_checks import (RoadGeometry, check_drivable_area, check_drivable_area_official,
    densify_trajectory, ego_body_polygon, first_frame_yaw, load_official_drivable,
    load_scene_road)
from run_ego_reference import DEFAULT_TARGET_TRACK, evaluate, polygon, scene_from_frames
from ego_editor import _track


@timed_operation('ego.independent_check', None)
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--target-track', required=True)
    parser.add_argument('--output', type=Path, help='New report path; preserve prior review')
    args = parser.parse_args()
    out_path = args.output if args.output is not None else args.run/'independent-check.json'
    if out_path.exists():
        raise FileExistsError(f'Refusing to overwrite: {out_path}')

    frames = json.loads((args.input/'metadata/frames.json').read_text())
    times, baseline_xy, baseline_yaw, actors = scene_from_frames(frames, args.target_track)
    xy, yaw = _track(args.run/'candidates'/args.candidate/'trajectory.csv', times)

    result = evaluate(times, xy, yaw, actors, times, args.target_track)
    dt = np.diff(times)
    motion = np.diff(xy, axis=0)/dt[:, None]
    midpoint = (yaw[:-1]+yaw[1:])/2
    lateral = np.abs(np.sum(motion*np.column_stack([-np.sin(midpoint), np.cos(midpoint)]), axis=1)).max()

    official = load_official_drivable(args.input/'metadata/map_drivable_official.json')
    road = load_scene_road(args.input, frames)
    config_path = args.run/'config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    source = {}
    if config.get('constructor_run'):
        source_path = Path(config['constructor_run'])/'reference_screen/summary.json'
        source = json.loads(source_path.read_text()) if source_path.is_file() else {}
        if source.get('road_source') == 'official_union':
            road = RoadGeometry(official.union, map_source=str(args.input/'metadata/map_drivable_official.json'),
                                polygon_count=len(official.polygons))
    saved = config.get('acceptance_policy') or source.get('acceptance_policy')
    effective = policy.resolve(saved) if saved and 'profile' in saved else None
    margin = policy.OFFICIAL_MARGIN_M if effective is None else effective['official_margin_m']
    strict_road = check_drivable_area(times, xy, yaw, road, tolerance_m=1e-8)
    official_result = check_drivable_area_official(times, xy, yaw, official, min_margin_m=margin)

    # densified linear-interpolation clearance against the recorded target (assumption)
    dense_t, dense_xy, dense_yaw, obs_frames = densify_trajectory(times, xy, yaw, max_step_s=0.1)
    target_series = [next(a for a in frame if a['track'] == args.target_track) for frame in actors]
    tx = np.array([a['x'] for a in target_series])
    ty = np.array([a['y'] for a in target_series])
    tyaw = np.unwrap([a['yaw'] for a in target_series])
    tl = np.array([a['length'] for a in target_series]); tw = np.array([a['width'] for a in target_series])
    gaps = []
    for i, t in enumerate(dense_t):
        j = max(0, min(int(np.searchsorted(times, t, side='right')-1), len(times)-2))
        w = float((t-times[j])/(times[j+1]-times[j]))
        bx, by = tx[j]*(1-w)+tx[j+1]*w, ty[j]*(1-w)+ty[j+1]*w
        byaw = tyaw[j]*(1-w)+tyaw[j+1]*w
        bl = tl[j]*(1-w)+tl[j+1]*w; bw = tw[j]*(1-w)+tw[j+1]*w
        gaps.append(ego_body_polygon(dense_xy[i, 0], dense_xy[i, 1], dense_yaw[i]).distance(
            polygon(bx, by, byaw, bl, bw)))
    from ego_action_tool import assess
    from ego_lane_markings import load_no_cross_lines, check_no_cross_lines
    from run_ego_reference import progress_with_safe_stop
    source = {}
    if config.get('constructor_run'):
        source = json.loads((Path(config['constructor_run'])/'reference_screen/summary.json').read_text())
    lines = load_no_cross_lines(args.input/'metadata/no_cross_lines.json')
    screened = assess(times, baseline_xy, baseline_yaw, actors, xy, yaw, 'retime',
        target_track=args.target_track, road=road, road_official=official,
        allow_safe_stop=bool(source.get('allow_safe_stop', False)), no_cross_lines=lines,
        official_min_margin_m=margin, acceptance=effective)
    initial_path = args.run/'candidates'/args.candidate/'initial-candidate.json'
    if initial_path.is_file() and args.candidate.startswith('constructor_'):
        record=next(row for row in source['candidates'] if 'constructor_'+row['candidate_id'].removeprefix('candidate_')==args.candidate)
        if record.get('operation') is not None:
            from ego_trajectory_edit import verify_braking_candidate,apply_curve_checks,apply_intent
            curve=verify_braking_candidate(times,baseline_xy,baseline_yaw,xy,yaw,record['operation'],effective['observation_end_s'])
            if curve != record['generation_curve']:
                raise ValueError('Constructor curve evidence differs from recomputation')
            apply_curve_checks(screened,curve)
            apply_intent(screened,times,xy,yaw,record['operation'])
    if initial_path.is_file() and args.candidate.startswith('foci_'):
        foci_run=Path(config['foci_run'])
        foci_config=json.loads((foci_run/'config.json').read_text())
        if foci_config.get('version') in ('ego_foci_probe.v5','ego_foci_probe.v6','ego_foci_probe.v7','ego_foci_probe.v8'):
            from run_ego_foci_probe import verify_candidate
            optimization=json.loads((foci_run/'candidates'/args.candidate/'optimization.json').read_text())
            parent_id='constructor_'+optimization.get('parent_candidate_id',foci_config['parent_candidate_id']).removeprefix('candidate_')
            px,pyaw=_track(args.run/'candidates'/parent_id/'trajectory.csv',times)
            fx,fyaw=verify_candidate(foci_run,args.candidate,input_dir=args.input,target_track=args.target_track,
                acceptance=effective,times=times,actors=actors,parent_xy=px,parent_yaw=pyaw)
            if not np.allclose(xy,fx,atol=1e-9,rtol=0) or not np.allclose(yaw,fyaw,atol=1e-9,rtol=0):
                raise ValueError('Initial FOCI candidate differs from optimizer replay')
    edit_path = args.run/'candidates'/args.candidate/'edit.json'
    if edit_path.is_file():
        from ego_trajectory_edit import apply_intent
        operation = json.loads(edit_path.read_text())
        if operation.get('mode')=='foci':
            from run_ego_foci_probe import verify_candidate
            run=Path(operation['foci_run']).resolve()
            if (args.run/'generation').resolve() not in run.parents:
                raise ValueError('FOCI source is outside this editor run')
            foci_config=json.loads((run/'config.json').read_text())
            if Path(foci_config['parent_run']).resolve()!=args.run.resolve() or foci_config['parent_candidate_id']!=operation['parent_candidate_id']:
                raise ValueError('FOCI imported parent identity differs')
            parent_xy,parent_yaw=_track(args.run/'candidates'/operation['parent_candidate_id']/'trajectory.csv',times)
            expected_xy,expected_yaw=verify_candidate(run,operation['foci_candidate_id'],
                input_dir=args.input,target_track=args.target_track,acceptance=effective,
                times=times,actors=actors,parent_xy=parent_xy,parent_yaw=parent_yaw)
            if not np.allclose(xy,expected_xy,atol=1e-9,rtol=0) or not np.allclose(yaw,expected_yaw,atol=1e-9,rtol=0):
                raise ValueError('FOCI imported candidate differs from replayed optimizer controls')
        if (edit_path.parent/'curve-check.json').is_file():
            from ego_trajectory_edit import modern_edit, apply_curve_checks
            def lookup(key):
                if key=='recorded_baseline': return baseline_xy,baseline_yaw
                return _track(args.run/'candidates'/key/'trajectory.csv',times)
            curve_cache={}
            def parent_curve(key):
                if key in curve_cache: return curve_cache[key]
                folder=args.run/'candidates'/key
                parent_operation=folder/'edit.json'
                if parent_operation.is_file() and (folder/'curve-check.json').is_file():
                    prior=json.loads(parent_operation.read_text())
                    expected_xy,expected_yaw,evidence=modern_edit(times,lookup,prior,effective['observation_end_s'],parent_curve=parent_curve(prior['parent_candidate_id']),acceptance=effective)
                    saved_xy,saved_yaw=lookup(key)
                    if not np.allclose(expected_xy,saved_xy,atol=1e-9,rtol=0) or not np.allclose(expected_yaw,saved_yaw,atol=1e-9,rtol=0):
                        raise ValueError('Parent generation curve differs from its saved track')
                elif key.startswith('constructor_'):
                    record=next(row for row in source['candidates'] if 'constructor_'+row['candidate_id'].removeprefix('candidate_')==key)
                    evidence=None
                    if record.get('operation') is not None:
                        from ego_trajectory_edit import verify_braking_candidate
                        saved_xy,saved_yaw=lookup(key)
                        evidence=verify_braking_candidate(times,baseline_xy,baseline_yaw,saved_xy,saved_yaw,record['operation'],effective['observation_end_s'])
                        if evidence!=record['generation_curve']: raise ValueError('Parent constructor curve differs from recorded evidence')
                else:
                    evidence=None
                curve_cache[key]=evidence
                return evidence
            reproduced_xy,reproduced_yaw,curve = modern_edit(times,lookup,operation,effective['observation_end_s'],parent_curve=parent_curve(operation['parent_candidate_id']),acceptance=effective)
            if not np.allclose(reproduced_xy,xy,atol=1e-9,rtol=0) or not np.allclose(reproduced_yaw,yaw,atol=1e-9,rtol=0):
                raise ValueError('Saved edit differs from reconstructed generation curve')
            stored_curve = json.loads((edit_path.parent/'curve-check.json').read_text())
            if any(curve.get(key) != value for key,value in stored_curve.items()):
                raise ValueError('Stored curve evidence differs from recomputation')
            apply_curve_checks(screened,curve)
        apply_intent(screened, times, xy, yaw, operation)
    checks = screened['checks']
    payload = dict(
        status='passed' if all(checks.values()) else 'failed',
        motion_intent=screened.get('motion_intent'),
        candidate_id=args.candidate,
        min_target_clearance_m=result['min_target_clearance_m'],
        strict_road_passed=strict_road['passed'], strict_road_tolerance_m=1e-8,
        min_road_margin_m=strict_road['min_margin_m'],
        strict_failing_frames=strict_road['failing_obs_frames'],
        official_road_passed=official_result['passed'],
        official_worst_corner_margin_m=official_result['worst_corner_margin_m'],
        official_min_margin_m=official_result['min_margin_m'],
        official_failing_frames=sorted({v['frame'] for v in official_result['violations']}),
        max_speed_mps=screened['max_speed_mps'], max_acceleration_mps2=screened['max_acceleration_mps2'],
        native_max_speed_mps=result['max_speed_mps'], native_max_acceleration_mps2=result['max_acceleration_mps2'],
        generation_curve=({k:v for k,v in screened['generation_curve'].items() if k not in {'continuous_state','speed_route'}} if screened.get('generation_curve') else None),
        max_lateral_speed_mps=float(lateral), min_other_clearance_m=result['min_other_clearance_m'],
        checks=checks, submit_status=screened['submit_status'], input_qualification=screened['input_qualification'],
        full_window_status=screened['full_window_status'], editing_status=screened['editing_status'],
        coverage=screened['coverage'], acceptance_policy=screened['acceptance_policy'],
        lane_compliance=screened['lane_compliance'],
        road_acceptance=screened['road'],
        strict_road_scope='Diagnostic with zero slack; excluded from acceptance. Production policy is road_acceptance.',
        linear_interpolation_min_gap_m=float(min(gaps)),
        linear_interpolation_max_step_s=0.1,
        interpolation_scope='assumption, not additional observed data',
        physical_avoidance='undetermined')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == '__main__':
    main()
