"""Shared speed/route intent acceptance and actionable local failure feedback."""
import math


class EditFeasibilityError(ValueError):
    def __init__(self,code,message,**limits):
        self.code=code;self.limits=limits
        super().__init__(code+': '+message)


def failure_feedback(error,args,boundary,end):
    text=str(error);code=getattr(error,'code',text.split(':',1)[0])
    guidance={
        'target_unreachable':'The requested change does not fit the available time or acceleration. Adjust the stated target or duration using the measured limits.',
        'route_exhausted':'The requested motion exceeds the known route. Reduce travel or explicitly select a different known route; no extrapolation is performed.',
        'route_unavailable':'The selected route cannot support this edit. Check its direction, geometric connection and available moving interval.',
        'state_incompatible':'The inherited state cannot be joined consistently. Choose a supported continuous parent or an explicit connection.',
        'retiming_unavailable':'This parent has no explicit reconstructable braking phase. Use timed speed targets instead.',
        'direction_change_requires_stop':'Add a zero-speed target before changing direction; use two zero targets when waiting is intended.',
        'immutable_history':'The edit would change the fixed observation period. Keep the requested change at or after the earliest editable time.'}
    if code not in guidance:return None
    limits=dict(earliest_editable_time_s=float(boundary),scene_end_s=float(end))
    limits.update(getattr(error,'limits',{}))
    retry=[]
    if code=='route_exhausted':
        retry.append(dict(change='Reduce the requested speed, offset or travel duration',condition='Retain the selected route'))
        if args.get('edit_mode')=='speed_schedule' and args.get('path_source','parent')=='parent':
            retry.append(dict(change='Set path_source=recorded_baseline',condition='Only when continuing on the known recorded route is intended; checks still apply'))
        if args.get('edit_mode')=='route_edit' and args.get('speed_policy','preserve_speed')=='preserve_speed':
            retry.append(dict(change='Consider speed_policy=preserve_progress',condition='Only if the resulting total-speed increase is acceptable'))
    return dict(status='preview' if args.get('preview_only') else 'not_feasible',candidate_created=False,feasible=False,
        error_code=code,error=text,error_summary=guidance[code],failure_kind='requested_goal_unreachable',
        feasibility_limits=limits,retry_options=retry,global_infeasibility_proven=False)


def intent_checks(curve):
    """Check declared edits independently of the stage's collision/road policy."""
    import numpy as np
    from scipy.interpolate import PPoly
    checks={};details={}
    speed=curve.get('speed_edit_summary');route=curve.get('route_edit_summary')
    if speed:
        errors=[abs(row['actual_speed_mps']-row['requested_speed_mps']) for row in speed['target_evidence']]
        checks['speed_targets']=bool(all(math.isfinite(v) and v<=1e-6 for v in errors))
        details['maximum_speed_target_error_mps']=max(errors,default=0.)
        preserved=speed.get('preservation_evidence')
        checks['stopping_target_preserved']=preserved is None or bool(preserved['error']<=preserved['tolerance'])
        if preserved is not None:details['preservation_evidence']=preserved
        hold=curve.get('hold_starts_at_s')
        if hold is not None:
            cont=curve['continuous_state'];p=PPoly(np.asarray(cont['xy_coefficients']),np.asarray(cont['times_s']))
            times=np.unique(np.r_[float(hold),p.x[p.x>=hold]])
            gap=float(np.max(np.linalg.norm(p(times)-p(float(hold)),axis=1)))
            velocity=float(np.max(np.linalg.norm(p(times,1),axis=1)))
            a=PPoly(np.asarray(cont['yaw_coefficients']),np.asarray(cont['times_s']))
            rotation=float(np.max(np.abs(a(times)-a(float(hold)))))
            yaw_rate=float(np.max(np.abs(a(times,1))))
            checks['stationary_hold']=bool(gap<=1e-6 and velocity<=1e-6 and rotation<=1e-6 and yaw_rate<=1e-6)
            details.update(maximum_hold_position_error_m=gap,maximum_hold_speed_mps=velocity,maximum_hold_yaw_change_rad=rotation)
    if route:
        checks['route_target']=bool(math.isfinite(route['target_offset_error_m']) and route['target_offset_error_m']<=1e-4)
        details['target_offset_error_m']=route['target_offset_error_m']
        if route.get('return_offset_error_m') is not None:
            checks['route_return']=bool(route['return_offset_error_m']<=1e-4)
    passed=bool(checks and all(checks.values()))
    return dict(passed=passed,checks=checks,details=details,declared='speed_targets' if speed else 'lateral_route_edit',
        declared_reverse=bool(curve.get('declared_reverse',False)),direction_sequence=curve.get('direction_sequence'),
        reference_point=curve['actual_start_state']['reference_point'],
        terminal_stopped=bool(np.linalg.norm(curve['terminal_velocity_xy_mps'])<=1e-6),terminal_state_method='reconstructed_generation_curve')


def apply_edit_contract(result,curve):
    """Same checks and peak semantics for attack, ego, previews and reviews."""
    evidence=intent_checks(curve);result['generation_curve']=curve;result['motion_intent']=evidence
    for key in ('max_speed_mps','max_acceleration_mps2'):
        if key in result:
            result.setdefault('native_'+key,result[key])
            result['generated_curve_'+key]=curve[key]
            result[key]=max(result[key],curve[key])
    result['acceleration_peak']=curve.get('acceleration_peak')
    for key in ('checks','full_window_checks','editing_checks'):
        if isinstance(result.get(key),dict):result[key]['edit_intent']=evidence['passed']
    for key in ('coverage','full_window_coverage','editing_coverage'):
        if isinstance(result.get(key),dict):result[key]['edit_intent']='passed' if evidence['passed'] else 'failed'
    if 'reference_screen_passed' in result:
        result['reference_screen_passed']=all(result['checks'].values())
        result['failed_checks']=[key for key,value in result['checks'].items() if not value]
    if not evidence['passed'] and 'status' in result:
        result.update(status='reference_screen_failed',editing_status='reference_screen_failed',full_window_status='reference_screen_failed',submit_status='rejected')
    return result
