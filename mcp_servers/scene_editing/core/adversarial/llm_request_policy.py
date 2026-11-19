"""Pure request-policy translation shared by the host and bundled editors."""
from copy import deepcopy
from os import getenv

DIALECTS = {'thinking', 'qwen', 'kimi'}
# 界面上只有低/中/高三档，这里翻译成各 dialect 自己的刻度
LEVEL_TO_EFFORT = {'thinking': {'low': 'low', 'medium': 'high', 'high': 'max'},
                   'qwen': {'low': 'low', 'medium': 'medium', 'high': 'xhigh'},
                   'kimi': {'low': 'low', 'medium': 'high', 'high': 'max'}}
EFFORTS = {'thinking': {'low', 'high', 'max'}, 'qwen': {'low', 'medium', 'xhigh'},
           'kimi': {'low', 'high', 'max'}}


def validate_request_policy(profile):
    managed = any(key in profile for key in ('thinking_mode', 'request_dialect', 'tool_choice_policy'))
    if not managed:
        return
    dialect = profile.get('request_dialect')
    if dialect not in DIALECTS:
        raise ValueError('request_dialect must be thinking, qwen or kimi')
    mode = profile.get('thinking_mode')
    if mode not in ('enabled', 'disabled'):
        raise ValueError('thinking_mode must be enabled or disabled')
    if dialect == 'kimi' and mode != 'enabled':
        raise ValueError('Kimi K3 does not support disabling thinking')
    effort = profile.get('reasoning_effort')
    if effort not in EFFORTS[dialect]:
        raise ValueError('Unsupported reasoning_effort for request_dialect ' + dialect)
    if profile.get('tool_choice_policy', 'preserve') not in ('preserve', 'auto'):
        raise ValueError('tool_choice_policy must be preserve or auto')
    if 'thinking_budget' in profile:
        raise ValueError('Managed effort profiles cannot also set thinking_budget')
    if mode == 'enabled' and profile.get('required_tool_thinking') == 'disabled':
        raise ValueError('Enabled thinking conflicts with required_tool_thinking=disabled')


def _effort_level(profile):
    """Per-call override wins over the global setting."""
    override = profile.get('reasoning_level_override')
    if override in LEVEL_TO_EFFORT['thinking']:
        return override
    level = getenv('LLM_REASONING_EFFORT')
    return level if level in LEVEL_TO_EFFORT['thinking'] else None


def effective_reasoning_effort(profile):
    """Effort level this request profile will actually send.

    A per-call override wins over the global setting; when neither resolves, the
    profile's own value is returned unchanged.
    """
    profile = profile or {}
    effort = profile.get('reasoning_effort')
    level = _effort_level(profile)
    dialect = profile.get('request_dialect')
    if level and dialect in LEVEL_TO_EFFORT:
        return LEVEL_TO_EFFORT[dialect][level]
    return effort


def apply_request_policy(profile, parameters):
    """Translate declared capabilities; narrow named tools before using auto.

    Tool schemas, eligibility and local native-call validation remain authoritative.
    Unmanaged legacy profiles retain the caller's existing parameters.
    """
    level = _effort_level(profile)
    if level and profile.get('request_dialect') in LEVEL_TO_EFFORT:
        profile = dict(profile, reasoning_effort=LEVEL_TO_EFFORT[profile['request_dialect']][level])
    validate_request_policy(profile)
    result = deepcopy(parameters)
    if 'thinking_mode' not in profile:
        return result
    if profile.get('max_tokens') is not None:
        result['max_tokens'] = profile['max_tokens']
    extra = dict(result.get('extra_body') or {})
    if 'thinking_budget' in extra:
        raise ValueError('Cannot combine reasoning_effort with thinking_budget')
    for key in ('thinking', 'enable_thinking', 'reasoning_effort'):
        extra.pop(key, None)
    result.pop('reasoning_effort', None)
    dialect, mode = profile['request_dialect'], profile['thinking_mode']
    if dialect == 'qwen':
        extra.update(enable_thinking=mode == 'enabled')
        if mode == 'enabled':
            extra['reasoning_effort'] = profile['reasoning_effort']
    elif dialect == 'thinking':
        extra['thinking'] = {'type': mode}
        if mode == 'enabled':
            result['reasoning_effort'] = profile['reasoning_effort']
    else:
        result['reasoning_effort'] = profile['reasoning_effort']
    if extra:
        result['extra_body'] = extra
    else:
        result.pop('extra_body', None)
    choice = result.get('tool_choice')
    if profile.get('tool_choice_policy') == 'auto' and result.get('tools'):
        if isinstance(choice, dict):
            name = choice.get('function', {}).get('name')
            selected = [tool for tool in result['tools'] if tool.get('function', {}).get('name') == name]
            if choice.get('type') != 'function' or len(selected) != 1:
                raise ValueError('Named tool must have exactly one declared definition')
            result['tools'] = selected
            result['tool_choice'] = 'auto'
        elif choice in ('required', 'auto', None):
            result['tool_choice'] = 'auto'
        result['parallel_tool_calls'] = False
    return result
