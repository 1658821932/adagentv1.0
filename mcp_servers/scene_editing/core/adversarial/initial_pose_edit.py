import math

import numpy as np


DEFAULT_MAX_TRANSLATION_M=30.
DEFAULT_MAX_YAW_CHANGE_RAD=math.pi/2
DEFAULT_MAX_SPEED_CHANGE_MPS=4.
POSE_ARGUMENTS = ('position_offset_m', 'yaw_mode', 'yaw_offset_rad', 'yaw_rad', 'initial_speed_mps')


def parent_seed(editor,parent):
    return (parent.get('initial_seed_xy',editor.seed_xy),parent.get('initial_seed_yaw',editor.seed_yaw),
        parent.get('initial_pose'))


def adjusted_seed(editor,parent,spec):
    seed_xy,seed_yaw,_=parent_seed(editor,parent)
    if spec is None:return seed_xy,seed_yaw,None
    if not isinstance(spec,dict):raise ValueError('initial pose adjustment must be an object')
    if set(spec).difference(POSE_ARGUMENTS):raise ValueError('Unsupported initial pose adjustment fields')
    offset=spec.get('position_offset_m')
    if not isinstance(offset,dict) or set(offset)!={'forward_m','left_m'}:
        raise ValueError('position_offset_m requires forward_m and left_m')
    values=np.asarray([offset['forward_m'],offset['left_m']],dtype=float)
    if any(type(value) not in (int,float) or not math.isfinite(value) for value in offset.values()):
        raise ValueError('Initial pose translation must be finite')
    max_translation=float(editor.constraints.get('initial_pose_max_translation_m',DEFAULT_MAX_TRANSLATION_M))
    translation=float(np.linalg.norm(values))
    if translation>max_translation+1e-9:
        raise ValueError(f'Initial pose translation exceeds {max_translation:g}m')
    original_delta=np.asarray(seed_xy[0])+values-np.asarray(editor.seed_xy[0])
    if np.linalg.norm(original_delta)>max_translation+1e-9:
        raise ValueError('Cumulative initial pose translation exceeds the configured bound')
    mode=spec.get('yaw_mode','preserve')
    if mode not in {'preserve','route_aligned','offset','explicit'}:
        raise ValueError('Unknown initial pose yaw_mode')
    original_yaw=float(seed_yaw[0])
    if mode=='preserve':
        if 'yaw_offset_rad' in spec or 'yaw_rad' in spec:raise ValueError('preserve yaw mode takes no yaw value')
        target_yaw=original_yaw
    elif mode=='route_aligned':
        if 'yaw_offset_rad' in spec or 'yaw_rad' in spec:raise ValueError('route_aligned takes no explicit yaw value')
        position=np.asarray(seed_xy[0])+values
        index=int(np.argmin(np.linalg.norm(editor.ego-position,axis=1)))
        target_yaw=float(editor.ego_yaw[index])
    elif mode=='offset':
        value=spec.get('yaw_offset_rad')
        if set(spec)-{'position_offset_m','yaw_mode','yaw_offset_rad','initial_speed_mps'} or type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('offset yaw mode requires finite yaw_offset_rad')
        target_yaw=original_yaw+float(value)
    else:
        value=spec.get('yaw_rad')
        if set(spec)-{'position_offset_m','yaw_mode','yaw_rad','initial_speed_mps'} or type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('explicit yaw mode requires finite yaw_rad')
        target_yaw=float(value)
    change=float(np.arctan2(np.sin(target_yaw-original_yaw),np.cos(target_yaw-original_yaw)))
    max_yaw=float(editor.constraints.get('initial_pose_max_yaw_change_rad',DEFAULT_MAX_YAW_CHANGE_RAD))
    if abs(change)>max_yaw+1e-9:raise ValueError('Initial pose yaw change exceeds the configured bound')
    cumulative=float(np.arctan2(np.sin(target_yaw-float(editor.seed_yaw[0])),np.cos(target_yaw-float(editor.seed_yaw[0]))))
    if abs(cumulative)>max_yaw+1e-9:raise ValueError('Cumulative initial pose yaw change exceeds the configured bound')
    rotation=np.asarray([[math.cos(change),-math.sin(change)],[math.sin(change),math.cos(change)]])
    adjusted_xy=(seed_xy-seed_xy[0])@rotation.T+seed_xy[0]+values
    adjusted_yaw=seed_yaw+change
    requested_speed=spec.get('initial_speed_mps')
    current_velocity=(adjusted_xy[1]-adjusted_xy[0])/(editor.times[1]-editor.times[0])
    current_speed=float(current_velocity@np.asarray([math.cos(adjusted_yaw[0]),math.sin(adjusted_yaw[0])]))
    if requested_speed is not None:
        if editor.constraints.get('stationary_target'):
            raise ValueError('A stationary target cannot change its zero initial speed')
        if type(requested_speed) not in (int,float) or not math.isfinite(requested_speed):
            raise ValueError('initial_speed_mps must be finite')
        import acceptance_policy as policy
        effective=policy.resolve(editor.constraints.get('acceptance_policy'))
        if not 0.<=requested_speed<=effective['generator_max_speed_mps']:
            raise ValueError('initial_speed_mps is outside the generator speed range')
        max_change=float(editor.constraints.get('initial_speed_max_change_mps',DEFAULT_MAX_SPEED_CHANGE_MPS))
        if abs(requested_speed-current_speed)>max_change+1e-9:
            raise ValueError('Initial speed change exceeds the configured bound')
        adjusted_xy=adjusted_xy.copy()
        adjusted_xy[1]=adjusted_xy[0]+float(requested_speed)*np.asarray(
            [math.cos(adjusted_yaw[0]),math.sin(adjusted_yaw[0])])*(editor.times[1]-editor.times[0])
    record=dict(schema_version='initial_pose_adjustment.v1',position_offset_m=dict(forward_m=float(values[0]),left_m=float(values[1])),
        translation_m=translation,yaw_mode=mode,yaw_change_rad=change,
        original_pose=dict(x_m=float(seed_xy[0][0]),y_m=float(seed_xy[0][1]),yaw_rad=original_yaw),
        adjusted_pose=dict(x_m=float(adjusted_xy[0][0]),y_m=float(adjusted_xy[0][1]),yaw_rad=float(adjusted_yaw[0])),
        coordinate_frame='scene-local initial ego x forward, y left; yaw counterclockwise')
    record['initial_speed_mps']=current_speed if requested_speed is None else float(requested_speed)
    record['source_initial_speed_mps']=current_speed
    record['initial_speed_changed']=requested_speed is not None
    return adjusted_xy,adjusted_yaw,record


def _feedback_reference(editor):
    rows=[]
    probe=getattr(editor,'policy_probe',None)
    results=getattr(probe,'results',{}) or {}
    for value in results.values():
        if isinstance(value,dict) and value.get('scene_feedback',{}).get('keyframes'):
            rows=value['scene_feedback']['keyframes']
    feedback=editor.constraints.get('revision_feedback') or {}
    evidence=feedback.get('evidence') if isinstance(feedback,dict) else None
    if isinstance(evidence,dict):
        scene=evidence.get('scene_feedback',evidence)
        if isinstance(scene,dict) and scene.get('keyframes'):rows=scene['keyframes']
    result=[]
    for row in rows:
        if not isinstance(row,dict) or 'ego_body_center_xy_m' not in row:continue
        result.append(dict(time_s=float(row['time_s']),xy=np.asarray(row['ego_body_center_xy_m'],float),
            yaw=float(row.get('ego_heading_rad',0.))))
    if result:return result
    from run_ego_reference import EGO_FRONT,EGO_REAR
    center=editor.ego+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(editor.ego_yaw),np.sin(editor.ego_yaw)])
    return [dict(time_s=float(t),xy=point,yaw=float(angle)) for t,point,angle in zip(editor.times,center,editor.ego_yaw)]


def search_specs(arguments,editor=None,parent=None):
    search=arguments.get('initial_pose_search')
    if search is not None:
        if arguments.get('preserve_outside_window'):
            raise ValueError('Initial pose search cannot preserve the parent outside the edit window')
        if not isinstance(search,dict):raise ValueError('initial_pose_search must be an object')
        allowed={'forward_offsets_m','left_offsets_m','yaw_mode','yaw_offset_rad','yaw_rad','initial_speed_mps'}
        if set(search)-allowed:raise ValueError('Unsupported initial_pose_search fields')
        forward=search.get('forward_offsets_m',[0.])
        left=search.get('left_offsets_m',[0.])
        if not isinstance(forward,list) or not isinstance(left,list) or not 1<=len(forward)<=4 or not 1<=len(left)<=4:
            raise ValueError('Initial pose offset axes must contain 1..4 values')
        if len(forward)*len(left)>6:raise ValueError('Initial pose search supports at most 6 combinations')
        shared={key:search[key] for key in ('yaw_mode','yaw_offset_rad','yaw_rad','initial_speed_mps') if key in search}
        return [dict(position_offset_m=dict(forward_m=f,left_m=l),**shared) for f in forward for l in left]
    if editor is None or parent is None:return [None]
    if arguments.get('preserve_outside_window'):
        raise ValueError('Unified construction cannot preserve the parent while changing initial state')
    candidates=[None]
    window=editor.constraints.get('event_window_s',[1.5,7.5])
    reference=[row for row in _feedback_reference(editor) if window[0]-.001<=row['time_s']<=window[1]+.001]
    if reference:
        indices=np.unique(np.linspace(0,len(reference)-1,min(2,len(reference)),dtype=int))
        for index in indices:
            row=reference[int(index)]
            target=np.asarray([np.interp(row['time_s'],editor.times,parent['xy'][:,axis]) for axis in range(2)])
            delta=row['xy']-target
            if np.linalg.norm(delta)<.5:continue
            candidates.append(dict(position_offset_m=dict(forward_m=float(delta[0]),left_m=float(delta[1])),yaw_mode='preserve'))
    for forward,left,yaw in [(-8.,0.,'preserve'),(-16.,0.,'preserve'),
            (-8.,2.,'preserve'),(0.,-2.,'preserve')]:
        candidates.append(dict(position_offset_m=dict(forward_m=forward,left_m=left),yaw_mode=yaw))
    if not editor.constraints.get('stationary_target'):
        seed_xy,seed_yaw,_=parent_seed(editor,parent)
        velocity=(seed_xy[1]-seed_xy[0])/(editor.times[1]-editor.times[0])
        speed=float(velocity@np.asarray([math.cos(seed_yaw[0]),math.sin(seed_yaw[0])]))
        import acceptance_policy as policy
        maximum=policy.resolve(editor.constraints.get('acceptance_policy'))['generator_max_speed_mps']
        for desired in dict.fromkeys([max(0.,round(speed-3.,3)),min(maximum,round(speed+2.,3))]):
            candidates.append(dict(position_offset_m=dict(forward_m=0.,left_m=0.),yaw_mode='preserve',initial_speed_mps=desired))
    result=[];seen=set()
    for spec in candidates:
        key=None if spec is None else (round(spec['position_offset_m']['forward_m'],3),
            round(spec['position_offset_m']['left_m'],3),spec.get('yaw_mode','preserve'),spec.get('initial_speed_mps'))
        if key in seen:continue
        try:adjusted_seed(editor,parent,spec)
        except (ValueError,TypeError,KeyError):continue
        seen.add(key);result.append(spec)
        if len(result)>=8:break
    return result or [None]


def adjusted_arrays(editor,parent,spec):
    from vehicle_control_search import infer_controls,integrate_controls
    seed_xy,seed_yaw,record=adjusted_seed(editor,parent,spec)
    static=np.array_equal(parent['dense_xy'],np.tile(parent['dense_xy'][0],(len(parent['dense_xy']),1)))
    if editor.constraints.get('stationary_target') or static:
        original=parent_seed(editor,parent)
        translation=seed_xy[0]-original[0][0]
        yaw_change=float(np.arctan2(np.sin(seed_yaw[0]-original[1][0]),np.cos(seed_yaw[0]-original[1][0])))
        rotation=np.asarray([[math.cos(yaw_change),-math.sin(yaw_change)],[math.sin(yaw_change),math.cos(yaw_change)]])
        dense_xy=(parent['dense_xy']-original[0][0])@rotation.T+seed_xy[0]
        dense_yaw=parent['dense_yaw']+yaw_change
        xy=(parent['xy']-original[0][0])@rotation.T+seed_xy[0]
        yaw=parent['yaw']+yaw_change
        return xy,yaw,dense_xy,dense_yaw,(seed_xy,seed_yaw,record)
    controls=parent['result'].get('control_sequence') or infer_controls(editor.dense_times,
        parent['dense_xy'],parent['dense_yaw'],editor.target['size_m'],editor.edit_start_s)
    arrays,_=integrate_controls(editor.times,seed_xy,seed_yaw,editor.dense_times,
        editor.target['size_m'],controls,editor.edit_start_s,acceptance=editor.constraints.get('acceptance_policy'))
    return *arrays,(seed_xy,seed_yaw,record)
