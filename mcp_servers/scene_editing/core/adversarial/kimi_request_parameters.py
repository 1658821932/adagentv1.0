"""Explicit provider capabilities and K3 compatibility; preserve tool restrictions."""


def normalize_kimi_parameters(model, parameters):
    result = dict(parameters)
    import os
    if os.environ.get('ADVERSARIAL_API_MAX_TOKENS'):
        result['max_tokens'] = int(os.environ['ADVERSARIAL_API_MAX_TOKENS'])
    if os.environ.get('ADVERSARIAL_THINKING_MODE'):
        from llm_request_policy import apply_request_policy
        profile = dict(model=model, thinking_mode=os.environ['ADVERSARIAL_THINKING_MODE'],
            request_dialect=os.environ.get('ADVERSARIAL_REQUEST_DIALECT'),
            reasoning_effort=os.environ.get('ADVERSARIAL_REASONING_EFFORT'),
            tool_choice_policy=os.environ.get('ADVERSARIAL_TOOL_CHOICE_POLICY','preserve'))
        if os.environ.get('ADVERSARIAL_API_MAX_TOKENS'):
            profile['max_tokens'] = int(os.environ['ADVERSARIAL_API_MAX_TOKENS'])
        return apply_request_policy(profile, result)
    mode = os.environ.get('ADVERSARIAL_REQUIRED_TOOL_THINKING', 'preserve')
    if mode not in ('preserve', 'disabled'):
        raise ValueError('ADVERSARIAL_REQUIRED_TOOL_THINKING must be preserve or disabled')
    choice = result.get('tool_choice')
    forced = choice == 'required' or (isinstance(choice, dict) and choice.get('type') == 'function')
    if mode == 'disabled' and forced:
        extra = dict(result.get('extra_body') or {})
        extra['thinking'] = {'type': 'disabled'}
        result['extra_body'] = extra
    if model != 'kimi-k3' and not model.startswith('kimi-k3.'):
        return result
    # K3 uses reasoning_effort, not the K2.x thinking field.
    extra = result.get('extra_body')
    if isinstance(extra, dict) and 'thinking' in extra:
        extra = {key:value for key,value in extra.items() if key != 'thinking'}
        if extra:
            result['extra_body'] = extra
        else:
            result.pop('extra_body')
    choice = result.get('tool_choice')
    if isinstance(choice, dict) and choice.get('type') == 'function':
        name = choice['function']['name']
        tools = [tool for tool in result.get('tools', []) if tool.get('function', {}).get('name') == name]
        if len(tools) != 1:
            raise ValueError('Named tool must have exactly one declared definition')
        # Named tool_choice conflicts with thinking. A single exposed tool plus
        # required preserves the same restriction while keeping K3 reasoning on.
        result.update(tool_choice='required', tools=tools)
    return result
