"""Current public editing tools; archived editing calls are not accepted."""
import copy

ALIASES = {'edit_route': 'route_edit', 'edit_speed_profile': 'speed_schedule'}
EDIT_TOOL_NAMES = set(ALIASES)

def validate_point_arguments(arguments):
    mode = arguments.get('edit_mode')
    if mode == 'route_edit':
        from route_edit_contract import validate
    elif mode == 'speed_schedule':
        from speed_profile_contract import validate
    else:
        raise ValueError('Only edit_route and edit_speed_profile are supported')
    validate(arguments)

def normalize_edit_call(name, arguments):
    if name not in ALIASES:
        raise ValueError('Unsupported editing tool: '+name)
    arguments = copy.deepcopy(arguments)
    if 'edit_mode' in arguments:
        raise ValueError(name+' already selects the mode; do not pass edit_mode')
    arguments['edit_mode'] = ALIASES[name]
    validate_point_arguments(arguments)
    return arguments

def mode_tools(template):
    from route_edit_contract import tool as route_tool
    from speed_profile_contract import tool as speed_tool
    return [route_tool(), speed_tool()]

def fields_match(value, required, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(label+' must be an object')
    missing, extra = sorted(required-set(value)), sorted(set(value)-allowed)
    if missing or extra:
        raise ValueError(f'{label}: missing_fields={missing}; unexpected_fields={extra}')
