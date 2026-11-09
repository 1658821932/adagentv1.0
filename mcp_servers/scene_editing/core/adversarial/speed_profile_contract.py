"""Dependency-free signed-speed tool schema and argument validation."""
import json
import math

EXAMPLE=dict(parent_candidate_id='candidate_001',operation='set_speed_targets',start_time_s=3.0,
    targets=[dict(time_s=7.0,speed_mps=0.0)],preview_only=True,
    reason='Preview a smooth stop while preserving the parent route and observation history.')

REQUIRED={'parent_candidate_id','operation','reason'}
OPTIONAL={'start_time_s','targets','path_source','transition_style','max_acceleration_mps2',
          'braking_start_shift_s','preserve','preview_only'}


def target_schema():
    return dict(type='array',minItems=1,maxItems=6,items=dict(type='object',
        properties=dict(time_s=dict(type='number'),speed_mps=dict(type='number')),
        required=['time_s','speed_mps'],additionalProperties=False))


def tool():
    props=dict(parent_candidate_id=dict(type='string'),
        operation=dict(type='string',enum=['set_speed_targets','retime_braking']),
        reason=dict(type='string',description='Benefit to retain, defect to change, and tradeoffs to check.'),
        start_time_s=dict(type='number',description='Required for set_speed_targets. Parent history through this time is fixed.'),
        targets=target_schema(),
        path_source=dict(type='string',enum=['parent','recorded_baseline'],description='Default parent. Uses known spatial route in both directions; never extrapolates. Recorded baseline permits original-route continuation with a checked connection.'),
        transition_style=dict(type='string',enum=['smooth','constant_acceleration'],description='For retime_braking, omission preserves the parent braking style. For new speed targets, default smooth eases acceleration in/out; constant_acceleration changes speed linearly. Smooth does not guarantee a lower peak. Equal zero targets wait.'),
        max_acceleration_mps2=dict(type='number',description='Optional tangential acceleration magnitude cap; cannot raise the existing hard planar limit.'),
        braking_start_shift_s=dict(type='number',description='Required for retime_braking. Positive delays the original braking start, negative advances it. Backend removes/rebuilds the original braking from the earliest affected time.'),
        preserve=dict(type='string',enum=['stop_position','stop_time'],description='For retime_braking only; required. Select which original stopping target to retain; other outcome is recalculated. Works only for a reconstructed explicit braking segment.'),
        preview_only=dict(type='boolean',description='Default false. True screens without creating a candidate or using a candidate slot; still uses a model action.'))
    return dict(type='function',function=dict(name='edit_speed_profile',description=(
        'Edit when/how fast to travel on a known route. set_speed_targets requires start_time_s and timed targets: '
        'positive=forward, zero=stop, negative=reverse WITHOUT flipping body heading. Supply a zero-speed target between forward and reverse; '
        'two zero targets mean wait. After the last target its speed is maintained to scene end. '
        'Supports braking, cruising, acceleration and restarting; no automatic stop requirement. '
        'retime_braking requires braking_start_shift_s and preserve=stop_position or stop_time; derives its affected start from the explicit parent braking curve. '
        'A sampled FOCI parent has no invented named braking segment: use timed targets instead. '
        'Parent route/history are retained; motion/road/contact checks decide eligibility. Preview is not a submittable candidate. '
        'Read decision_summary first; recommended_previews are conditional complete calls, not automatic changes. '
        'Minimal preview example: '+json.dumps(EXAMPLE,separators=(',',':'))),
        parameters=dict(type='object',properties=props,required=sorted(REQUIRED),additionalProperties=False)))


def number(value,label):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(label+' must be a finite number')
    return float(value)


def validate_targets(targets, *, start=None, end=None, max_speed=None):
    if not isinstance(targets,list) or not 1<=len(targets)<=6:
        raise ValueError('targets requires 1..6 timed speed states')
    previous=start
    for point in targets:
        if not isinstance(point,dict) or set(point)!={'time_s','speed_mps'}:
            raise ValueError('Each target requires exactly time_s and speed_mps')
        t=number(point['time_s'],'target.time_s');v=number(point['speed_mps'],'target.speed_mps')
        if previous is not None and t<=previous:
            raise ValueError('Target times must strictly increase after the edit start')
        if end is not None and t>end+1e-8:
            raise ValueError('Target time exceeds scene end')
        if max_speed is not None and abs(v)>max_speed:
            raise ValueError('Target signed speed exceeds unchanged hard speed limit')
        previous=t


def validate(args):
    from ego_edit_contract import fields_match
    fields_match(args,REQUIRED,REQUIRED|OPTIONAL|{'edit_mode'},'edit_speed_profile')
    if not isinstance(args['parent_candidate_id'],str) or not args['parent_candidate_id']:
        raise ValueError('An existing parent candidate is required')
    if not isinstance(args['reason'],str) or not args['reason'].strip():
        raise ValueError('reason must explain what to preserve and change')
    op=args['operation']
    if op=='set_speed_targets':
        if not {'start_time_s','targets'}<=set(args) or {'braking_start_shift_s','preserve'}&set(args):
            raise ValueError('set_speed_targets requires start_time_s and targets; omit retiming fields')
        validate_targets(args['targets'])
    elif op=='retime_braking':
        if not {'braking_start_shift_s','preserve'}<=set(args) or {'start_time_s','targets'}&set(args):
            raise ValueError('retime_braking requires braking_start_shift_s and preserve; omit start_time_s/targets')
        if args['preserve'] not in ('stop_position','stop_time'):
            raise ValueError('preserve must be stop_position or stop_time')
    else:raise ValueError('Unknown speed operation')
    if args.get('path_source','parent') not in ('parent','recorded_baseline'):
        raise ValueError('path_source must be parent or recorded_baseline')
    if args.get('transition_style','smooth') not in ('smooth','constant_acceleration'):
        raise ValueError('Unknown transition_style')
    if 'preview_only' in args and type(args['preview_only']) is not bool:
        raise ValueError('preview_only must be boolean')
    for k in ('start_time_s','braking_start_shift_s','max_acceleration_mps2'):
        if k in args:number(args[k],k)


