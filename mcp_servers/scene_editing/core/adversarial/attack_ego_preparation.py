"""Deterministic stage A: prepare and freeze ego acceleration before editing an actor."""
import json
from pathlib import Path

import numpy as np

from ego_history import TAKEOVER_S, clone, prepare, source_scene, verify
from ego_trajectory_edit import start_state
from experiment_runtime import atomic_json
from run_ego_reference import ego_path_from_frames, EGO_FRONT, EGO_REAR
from workflow_timing import timed_operation


def history_state(scene):
    frames=json.loads((Path(scene)/'metadata/frames.json').read_text())
    times,xy,yaw=ego_path_from_frames(frames)
    inherited=source_scene(scene)
    original_frames=(frames if inherited==Path(scene).resolve() else
        json.loads((inherited/'metadata/frames.json').read_text()))
    original_times,original_xy,original_yaw=ego_path_from_frames(original_frames)
    start=start_state(original_times,original_xy,original_yaw,0.)
    index=int(np.searchsorted(times,TAKEOVER_S+.001,side='right'))-1
    step=round(float(np.median(np.diff(times)))/.05)*.05
    centers=xy+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(yaw),np.sin(yaw)])
    if index<1:raise ValueError('Source does not cover the common 1.5s takeover')
    return dict(frozen=True,end_s=TAKEOVER_S,initial_speed_mps=start['signed_speed_mps'],
        takeover=dict(time_s=TAKEOVER_S,rear_axle_xy_m=xy[index].tolist(),yaw_rad=float(yaw[index]),
            body_center_xy_m=centers[index].tolist(),
            speed_mps=float(np.linalg.norm((centers[index]-centers[index-1])/step))),
        future_reference_scope='Route/reference only. VAD chooses its own actions after takeover.')


@timed_operation('attack.ego_history_prepare','output')
def prepare_acceleration(scene,output,acceptance,*,enabled=True,target_speed_mps=None,previous_scene=None,route=None):
    """No LLM or policy call. Explicit targets are never silently weakened."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=source_scene(scene)
    atomic_json(output/'input.json',dict(source_scene=str(source),enabled=enabled,
        target_speed_mps=target_speed_mps,route=route,previous_scene=None if previous_scene is None else str(previous_scene)))
    report=dict(stage='ego_history_prepare',attempts=[],api_calls=0,policy_tests=0,
        report_path=str(output/'report.json'))

    def finish(status,chosen,reason):
        report.update(status=status,scene=str(chosen),reason=reason,shared_state=history_state(chosen))
        atomic_json(output/'report.json',report)
        return report

    if previous_scene is None and enabled and target_speed_mps is None and route is None and (Path(scene)/'ego-history.json').is_file():
        previous_scene=scene
    if previous_scene is not None:
        chosen=clone(previous_scene,output/'scene')
        verify(chosen)
        return finish('reused',chosen,'Preserved the exact history of the checkpoint; no second acceleration.')
    if not enabled:
        return finish('unchanged',source,'Ego history preparation was explicitly disabled.')
    if route is not None:
        report['route'] = route
        if target_speed_mps is not None:
            return finish('unavailable',source,'Specify ego route end_speed_mps inside the route request only.')
        try:
            chosen = prepare(source, output/'route', None, acceptance, route=route)
            verify(chosen)
        except ValueError as error:
            report['attempts'].append(dict(status='rejected', reason=str(error)))
            return finish('unavailable',source,str(error))
        report['attempts'].append(dict(status='prepared', mode='endpoint_route'))
        return finish('prepared',chosen,'Fitted and froze the fixed-start ego route through 1.5s.')
    if target_speed_mps is not None and (type(target_speed_mps) not in (int,float) or not np.isfinite(target_speed_mps)):
        return finish('unavailable',source,'Explicit ego speed target must be a finite number.')
    state=history_state(source);initial=float(state['initial_speed_mps'])
    if initial<0:
        if target_speed_mps is not None:
            return finish('unavailable',source,'Forward acceleration preparation cannot inherit reversing motion.')
        return finish('unchanged',source,'Forward acceleration preparation is unavailable for a reversing source.')
    # A soft preparation preference from the existing research profile, not a
    # new acceptance threshold, scene-specific target or maximum-danger claim.
    preferred_peak=min(float(acceptance['comfort_reference_mps2']),float(acceptance['max_acceleration_mps2']))
    baseline_speed=float(state['takeover']['speed_mps'])
    desired=(min(float(acceptance['max_speed_mps']),max(initial,baseline_speed)+preferred_peak*TAKEOVER_S/1.5)
        if target_speed_mps is None else float(target_speed_mps))
    report.update(requested_target_speed_mps=desired,baseline_takeover_speed_mps=baseline_speed,preferred_peak_acceleration_mps2=preferred_peak,
        target_source='profile_acceleration_preference' if target_speed_mps is None else 'explicit_target')
    if not np.isfinite(desired) or desired < 0.:
        return finish('unavailable',source,'Ego speed target must be finite and nonnegative.')
    if target_speed_mps is None and desired<=initial+1e-8:
        return finish('unchanged',source,'The inherited speed already reaches the requested/preferred speed.')
    for number in range(1,2 if target_speed_mps is not None else 4):
        if target_speed_mps is None and desired<=baseline_speed+1e-8:break
        targets=[dict(time_s=TAKEOVER_S,speed_mps=desired)]
        try:
            chosen=prepare(source,output/f'trial_{number:02d}',targets,acceptance)
            verify(chosen)
        except ValueError as error:
            report['attempts'].append(dict(target_speed_mps=desired,status='rejected',reason=str(error)))
            if target_speed_mps is not None:
                return finish('unavailable',source,'Explicit ego speed target is infeasible; inspect the preparation report.')
            desired=(initial+desired)/2
            continue
        report['attempts'].append(dict(target_speed_mps=desired,status='prepared'))
        report['prepared_target_speed_mps']=desired
        if target_speed_mps is None and history_state(chosen)['takeover']['speed_mps']<=baseline_speed+1e-8:
            return finish('unchanged',source,'The feasible preparation did not increase sampled takeover speed; source history is retained.')
        return finish('prepared',chosen,'Generated and froze a feasible speed history before actor selection.')
    return finish('unchanged',source,'No tested acceleration passed the existing checks; source history is retained.')
