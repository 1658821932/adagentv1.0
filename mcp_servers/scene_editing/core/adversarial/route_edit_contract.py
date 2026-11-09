"""One goal-oriented route edit; lateral direction follows the forward route tangent."""
import json
import math

EXAMPLE=dict(parent_candidate_id='candidate_001',start_time_s=3.0,left_offset_m=-0.3,
    transition_duration_s=2.0,hold_duration_s=0.5,return_duration_s=2.0,
    end_behavior='return_to_route',speed_policy='preserve_speed',preview_only=True,
    reason='Preview a gentle right avoidance and return while preserving total speed.')

REQUIRED={'parent_candidate_id','start_time_s','left_offset_m','transition_duration_s','end_behavior','reason'}
OPTIONAL={'hold_duration_s','return_duration_s','speed_policy','preview_only'}

def validate(args):
    clean={k:v for k,v in args.items() if k not in {'edit_mode'}}
    if REQUIRED-set(clean) or set(clean)-REQUIRED-OPTIONAL:
        raise ValueError('edit_route fields: missing='+str(sorted(REQUIRED-set(clean)))+' unexpected='+str(sorted(set(clean)-REQUIRED-OPTIONAL)))
    for key in ('parent_candidate_id','reason'):
        if not isinstance(clean[key],str) or not clean[key].strip():raise ValueError(key+' must be nonempty')
    for key in ('start_time_s','left_offset_m','transition_duration_s','hold_duration_s','return_duration_s'):
        if key in clean and (type(clean[key]) not in (int,float) or not math.isfinite(clean[key])):raise ValueError(key+' must be finite')
    if clean['transition_duration_s']<=0 or clean.get('hold_duration_s',0)<0:raise ValueError('Transition duration must be positive; hold duration nonnegative')
    if clean['end_behavior'] not in ('keep_offset','return_to_route'):raise ValueError('end_behavior must be keep_offset or return_to_route')
    if clean['end_behavior']=='return_to_route':
        if clean.get('return_duration_s',0)<=0:raise ValueError('return_to_route requires positive return_duration_s')
    elif 'return_duration_s' in clean or 'hold_duration_s' in clean:raise ValueError('keep_offset has no hold/return durations; offset continues to scene end')
    if clean.get('speed_policy','preserve_speed') not in ('preserve_speed','preserve_progress'):raise ValueError('speed_policy must be preserve_speed or preserve_progress')
    if 'preview_only' in clean and type(clean['preview_only']) is not bool:raise ValueError('preview_only must be boolean')

def tool():
    props={
      'parent_candidate_id':{'type':'string'},'start_time_s':{'type':'number'},
      'left_offset_m':{'type':'number','description':'Signed offset from parent geometric route: positive left, negative right. Backend checks road/body clearance; no guessed steering angle.'},
      'transition_duration_s':{'type':'number','exclusiveMinimum':0},
      'end_behavior':{'type':'string','enum':['keep_offset','return_to_route'],'description':'keep_offset follows a parallel route until scene end. return_to_route removes offset after transition + hold + return; it returns to the spatial route, not necessarily the parent position at the same time.'},
      'hold_duration_s':{'type':'number','minimum':0},'return_duration_s':{'type':'number','exclusiveMinimum':0},
      'speed_policy':{'type':'string','enum':['preserve_speed','preserve_progress'],'description':'Default preserve_speed spends some forward progress on lateral motion and rejects lateral rates above available speed. preserve_progress retains parent along-route travel and may raise total speed. No automatic slowing; edit_speed_profile can prepare/repair speed explicitly.'},
      'preview_only':{'type':'boolean'},'reason':{'type':'string'}}
    description=('Smooth left/right avoidance and optional return, inheriting parent pose/velocity and unchanged observation history. '
        'Specify offset and durations; heading/curvature are computed. Forward parent routes only. Preview spends no candidate slot. '
        'Read decision_summary first; recommended_previews are conditional complete calls, not automatic changes. '
        'Minimal preview example: '+json.dumps(EXAMPLE,separators=(',',':')))
    return {'type':'function','function':{'name':'edit_route','description':description,'parameters':{'type':'object','properties':props,'required':sorted(REQUIRED),'additionalProperties':False}}}
