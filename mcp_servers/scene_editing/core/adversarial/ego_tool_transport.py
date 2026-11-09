"""Constrain allowed functions locally while using compatible auto transport."""
from copy import deepcopy


def constrained_auto(choice, tools):
    if not isinstance(choice, dict):
        return choice, tools
    name = choice['function']['name']
    selected = [deepcopy(tool) for tool in tools if tool['function']['name'] == name]
    if len(selected) != 1:
        raise ValueError('Forced phase function is missing or ambiguous: '+name)
    return 'auto', selected


def validate_exposed_call(name, arguments, tools):
    schema = next((tool['function']['parameters'] for tool in tools if tool['function']['name'] == name), None)
    if schema is None:
        raise ValueError('Tool was not exposed for this phase: '+name)
    if not set(schema.get('required', [])) <= set(arguments):
        raise ValueError('Missing required tool arguments')
    properties = schema.get('properties', {})
    if schema.get('additionalProperties') is False and set(arguments)-set(properties):
        raise ValueError('Unexpected tool arguments')
    for key, spec in properties.items():
        if key in arguments and 'enum' in spec and arguments[key] not in spec['enum']:
            raise ValueError('Argument is outside the phase whitelist: '+key)
