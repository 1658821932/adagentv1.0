"""Coarse alternatives guide object selection without becoming acceptance gates."""
import contextvars
from copy import deepcopy
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import time
from attack_budget import AUTO_SESSION_ROOT

_CLOCK=contextvars.ContextVar('attack_auto_stage_clock',default=None)


def shared_auto_session(function):
    signature=inspect.signature(function)
    @wraps(function)
    def wrapped(*args,**kwargs):
        values=signature.bind(*args,**kwargs).arguments
        if not (values.get('constraints') or {}).get('plan_options') or _CLOCK.get() is not None:
            return function(*args,**kwargs)
        session=Path(values['output'])/'plan-session-budget'
        if values.get('continue_from'):
            previous=Path(values['continue_from'])/'input.json'
            if previous.is_file():
                saved=json.loads(previous.read_text()).get('auto_session_root')
                if saved:
                    from experiment_runtime import experiments_root
                    allowed=Path(os.getenv('ADVERSARIAL_CONTINUATION_ROOT',str(experiments_root()))).resolve()
                    if not Path(saved).resolve().is_relative_to(allowed):
                        raise ValueError('Prior auto-selection budget is outside the continuation area')
                    session=Path(saved)
        root=AUTO_SESSION_ROOT.set(str(session))
        clock=_CLOCK.set({'deadline':None})
        try:return function(*args,**kwargs)
        finally:
            _CLOCK.reset(clock);AUTO_SESSION_ROOT.reset(root)
    return wrapped


def editor_deadline(seconds):
    clock=_CLOCK.get()
    if clock is None:return time.perf_counter()+seconds
    if clock['deadline'] is None:clock['deadline']=time.perf_counter()+seconds
    return clock['deadline']


def exclude_probe_wait(seconds):
    """Pause only the editor clock; probe count and external tool timeout still apply."""
    clock=_CLOCK.get()
    if clock is not None and clock['deadline'] is not None:
        clock['deadline']+=seconds
    return seconds


def selection_seconds(default):
    clock=_CLOCK.get()
    if clock is None or clock['deadline'] is None:return default
    return max(0.,min(default,clock['deadline']-time.perf_counter()))


def reselection_limit(constraints):
    return 2 if constraints.get('plan_options') else 1


def selection_tools(tools,assets,required_operation,required_track,options):
    result=[]
    for tool in deepcopy(tools):
        name=tool['function']['name']
        if name=='add_vehicle' and (not assets or required_operation=='edit_existing' or required_track is not None):continue
        if name=='select_vehicle' and required_operation=='add_vehicle':continue
        if options and name in ('select_vehicle','add_vehicle'):
            tool['function']['parameters']['properties']['option_id']=dict(type='string',
                enum=[option['option_id'] for option in options],
                description='Optional associated coarse suggestion; explain any adaptation in reason.')
        result.append(tool)
    return result


def choice(options,arguments,target):
    option_id=arguments.get('option_id')
    option=next((item for item in options if item['option_id']==option_id),None)
    if option_id is not None and option is None:raise ValueError('Unknown coarse option_id')
    adapted=bool(option and (option['operation']!=target['operation'] or
        option.get('target_track') not in (None,target['track_id'])))
    return dict(option_id=option_id,operation=target['operation'],target_track=target['track_id'],
        adapted=adapted,reason=arguments['reason'],suggestion=deepcopy(option),
        scope='Agent choice; coarse alternatives are suggestions, not hard task contracts')
