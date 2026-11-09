"""Structured Agent requests for bounded defense construction and FOCI."""
import math

NAMES = frozenset(('construct_defense_candidates', 'optimize_defense_candidate'))
NUMBER = dict(type='number')
INTERVAL = dict(type='array', items=NUMBER, minItems=2, maxItems=2)


def _tool(name, description, properties):
    return dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=properties, required=list(properties), additionalProperties=False)))


TOOLS = [
    _tool('construct_defense_candidates',
        'Construct bounded speed-reduction candidates. Search the requested time/deceleration ranges; '
        'preserve observation history. All created candidates, including failed checks, remain available. '
        'Positive target speed requires cruise; zero requires hold. This tool shares the new-candidate '
        'budget with edits, repairs and FOCI. No acceptance limit can be changed.',
        dict(parent_trajectory_id=dict(type='string'), action=dict(type='string', enum=['reduce_speed']),
             start_time_range_s=INTERVAL, target_speed_mps=NUMBER, deceleration_range_mps2=INTERVAL,
             end_behavior=dict(type='string', enum=['hold','cruise']),
             max_candidates=dict(type='integer', minimum=1, maximum=3))),
    _tool('optimize_defense_candidate',
        'Run FOCI with shared route-progress, reserve-shortfall and actual-motion objectives on this parent. '
        'History and later outside-window samples stay fixed; a window ending at scene end permits terminal adjustment. '
        'Optional preserve_route=true changes timing along the parent route instead of steering. '
        'terminal_behavior=stop smoothly approaches a hold over the final sampled interval (window must end at scene end). '
        'improvement_goal selects balanced, recover_progress or smoother_motion; safety reserve is retained in every case. '
        'Keep the parent and all checked outputs. Shares the new-candidate budget; execution errors are separate.',
        dict(parent_candidate_id=dict(type='string'), edit_window_s=INTERVAL,
             max_candidates=dict(type='integer', minimum=1, maximum=3))),
]

FOCI_OPTIONS = dict(preserve_route=dict(type='boolean',description='Default false. True keeps the parent spatial route and changes its timing; cannot extend beyond that route.'),
    terminal_behavior=dict(type='string',enum=['free','stop'],description='Default free. stop holds the final sampled interval and smooths entry over the preceding interval; final checks still apply.'),
    improvement_goal=dict(type='string',enum=['balanced','recover_progress','smoother_motion'],description='Default balanced. Search emphasis, not a guarantee or acceptance override.'))
FOCI_OPTIONS['speed_targets']=dict(type='array',minItems=1,maxItems=6,items=dict(type='object',properties=dict(time_s=NUMBER,speed_mps=NUMBER),required=['time_s','speed_mps'],additionalProperties=False),description='Optional timed signed speeds, same meanings as edit_speed_profile. Soft sample-based optimization preferences, not exact states or acceptance overrides. Zero=stop, negative=reverse without flipping body heading. Check returned speed_target_feedback.')
TOOLS[1]['function']['parameters']['properties'].update(FOCI_OPTIONS)


def normalize_request(name, args, end, rounding_s=.001):
    """Only normalize a rounded scene endpoint; never move the observation boundary."""
    result=dict(args)
    key='start_time_range_s' if name=='construct_defense_candidates' else 'edit_window_s'
    values=result.get(key)
    if isinstance(values,list):
        result[key]=[float(end) if type(value) in (int,float) and math.isfinite(value)
                     and abs(value-end)<=rounding_s else value for value in values]
    if name=='optimize_defense_candidate' and isinstance(result.get('speed_targets'),list):
        result['speed_targets']=[dict(p,time_s=float(end)) if isinstance(p,dict) and type(p.get('time_s')) in (int,float) and math.isfinite(p['time_s']) and abs(p['time_s']-end)<=rounding_s else p for p in result['speed_targets']]
    return result


def validate_request(name, args, *, boundary, end, remaining):
    spec=next((t['function']['parameters'] for t in TOOLS if t['function']['name']==name),None)
    if spec is None or not isinstance(args,dict) or not set(spec['required'])<=set(args)<=set(spec['properties']):
        raise ValueError('Generation request fields must exactly match the exposed schema')
    count=args['max_candidates']
    if type(count) is not int or not 1<=count<=min(3,remaining):
        raise ValueError('max_candidates exceeds the remaining shared new-candidate budget')
    key='start_time_range_s' if name=='construct_defense_candidates' else 'edit_window_s'
    interval=args[key]
    if (not isinstance(interval,list) or len(interval)!=2
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in interval)
            or not boundary<=interval[0]<=interval[1]<=end
            or (name=='optimize_defense_candidate' and interval[0]==interval[1])
            or (name=='construct_defense_candidates' and interval[1]>=end)):
        raise ValueError(key+' must lie after observation and inside the scene')
    parent_key='parent_trajectory_id' if name=='construct_defense_candidates' else 'parent_candidate_id'
    if not isinstance(args[parent_key],str) or not args[parent_key]:
        raise ValueError('An existing parent identity is required')
    if name=='construct_defense_candidates':
        strength=args['deceleration_range_mps2'];speed=args['target_speed_mps']
        if (args['action']!='reduce_speed' or not isinstance(strength,list) or len(strength)!=2
                or any(type(x) not in (int,float) or not math.isfinite(x) for x in strength)
                or not 0<strength[0]<=strength[1]
                or type(speed) not in (int,float) or not math.isfinite(speed) or speed<0
                or args['end_behavior']!=('hold' if speed==0 else 'cruise')):
            raise ValueError('Require positive ordered deceleration range and compatible target speed/end behavior')
    if name=='optimize_defense_candidate':
        if 'preserve_route' in args and type(args['preserve_route']) is not bool:
            raise ValueError('preserve_route must be boolean')
        if args.get('terminal_behavior','free') not in ('free','stop') or args.get('improvement_goal','balanced') not in ('balanced','recover_progress','smoother_motion'):
            raise ValueError('Unknown FOCI terminal behavior or improvement goal')
        if args.get('terminal_behavior')=='stop' and abs(interval[1]-end)>1e-8:
            raise ValueError('terminal_behavior=stop needs a window ending at scene end')
    if name=='optimize_defense_candidate' and 'speed_targets' in args:
        targets=args['speed_targets']
        if not isinstance(targets,list) or not 1<=len(targets)<=6:
            raise ValueError('speed_targets requires 1..6 timed states')
        previous=interval[0]
        for point in targets:
            if (not isinstance(point,dict) or set(point)!={'time_s','speed_mps'}
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in point.values())
                    or not previous<point['time_s']<=interval[1]):
                raise ValueError('speed_targets must be finite signed speeds at increasing times inside edit_window_s')
            previous=point['time_s']
    return count


def select_foci_parent(candidates):
    """Prefer valid, modest-acceleration reference paths; failures remain repair material."""
    if not candidates:
        return None
    def rank(row):
        checks=row.get('shared_checks',{})
        failures=sum(not value for value in checks.values())
        curve=row.get('generation_curve') or {}
        acceleration=max(row['max_acceleration_mps2'],curve.get('max_acceleration_mps2',0.))
        return (not row['reference_screen_passed'], failures, acceleration,
                abs((row.get('operation') or {}).get('lateral_offset_m',0.)))
    return min(candidates,key=rank)['candidate_id']


def select_foci_parents(candidates,limit=2):
    """Cover distinct checked progress/comfort branches within a fixed output budget."""
    valid=[r for r in candidates if r['reference_screen_passed'] and r.get('behavior_metrics')]
    if not valid:
        parent=select_foci_parent(candidates)
        return [] if parent is None else [parent]
    # Optimization costs allocate generation effort, never recommend a final candidate.
    first=min(valid,key=lambda r:r['behavior_metrics']['optimization_cost'])
    result=[first]
    while len(result)<min(limit,len(valid)):
        remaining=[r for r in valid if r not in result]
        def diversity(row):
            m=row['behavior_metrics']
            return min(abs(m['net_reference_progress_fraction']-p['behavior_metrics']['net_reference_progress_fraction'])+
                .05*abs(row['max_acceleration_mps2']-p['max_acceleration_mps2']) for p in result)
        result.append(max(remaining,key=diversity))
    return [r['candidate_id'] for r in result]
