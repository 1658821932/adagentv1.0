"""Scene-derived braking proposals and replay; no additional acceptance limits."""
import numpy as np
import acceptance_policy as policy


def proposals(times,xy,yaw,diagnostic,effective,allow_stop):
    boundary=policy.edit_start(effective)
    if not allow_stop or boundary>=times[-1]:return []
    from edit_continuity import parent_motion,state_at
    lookup=lambda key:(xy,yaw)
    try:
        curve=parent_motion(times,lookup,'recorded_baseline',None,boundary,boundary)
        state=state_at(times,xy,yaw,boundary,curve)
    except ValueError:
        # Extra proposals must not prevent the original pool from reporting
        # unsupported source motion through its existing checks.
        return []
    if state['signed_speed_mps']<=0:return []
    speed=float(np.linalg.norm(state['velocity_xy_mps']))
    if speed<.1:return []
    gap=diagnostic.get('edit_start_clearance_m')
    if gap is None or gap<=0:return []
    # Curvature only informs proposals. Delivered planar extrema remain decisive.
    velocity=np.asarray(state['velocity_xy_mps']);acc=np.asarray(state.get('acceleration_xy_mps2',[0.,0.]))
    normal=abs(float(velocity[0]*acc[1]-velocity[1]*acc[0]))/speed
    limit=effective['max_acceleration_mps2']
    cap=float(np.sqrt(max(0.,limit**2-normal**2)))
    if cap<=0:return []
    need=speed**2/(2*gap)
    # Explore both the contact boundary and curvature-adjusted strongest stop.
    # No candidate is declared feasible by this scalar estimate.
    strengths=sorted(set(min(cap*.995,max(.1,value)) for value in
        (need*.98,need*1.04,(need+cap)/2,cap*.97,cap*.995)))
    result=[]
    for a in strengths:
        end=boundary+speed/a
        if end>times[-1]:continue
        result.append(('speed_schedule',boundary,dict(operation='set_speed_targets',
            targets=[dict(time_s=float(end),speed_mps=0.)],transition_style='constant_acceleration',
            max_acceleration_mps2=limit,path_source='parent')))
    return result


def verify_constructor(times,xy,yaw,candidate_xy,candidate_yaw,operation,boundary,effective=None):
    from ego_trajectory_edit import verify_braking_candidate,modern_edit
    if operation.get('edit_mode')!='speed_schedule':
        return verify_braking_candidate(times,xy,yaw,candidate_xy,candidate_yaw,operation,boundary)
    lookup=lambda key:(xy,yaw) if key=='recorded_baseline' else (_ for _ in ()).throw(ValueError('Unknown constructor reference'))
    expected,angles,evidence=modern_edit(times,lookup,operation,boundary,acceptance=effective)
    if not np.allclose(expected,candidate_xy,atol=1e-9,rtol=0) or not np.allclose(angles,candidate_yaw,atol=1e-9,rtol=0):
        raise ValueError('Constructor speed schedule differs from reconstructed operation')
    return evidence


def braking_result(record):
    curve=record.get('generation_curve') or {}
    segment=curve.get('braking_segment') or {}
    return dict(stop_time_s=segment.get('stop_time_s',curve.get('hold_starts_at_s')),
        stop_rear_axle_xy_m=segment.get('stop_xy_m'),
        min_body_clearance_m=record.get('min_target_clearance_m'),
        native_peak_acceleration_mps2=record.get('max_acceleration_mps2'),
        generated_peak_acceleration_mps2=curve.get('max_acceleration_mps2'),
        failed_checks=[key for key,value in record.get('shared_checks',{}).items() if value is False],
        scope='Checked full-scene body clearance and existing motion limits; not target-centre distance or a physical certificate')
