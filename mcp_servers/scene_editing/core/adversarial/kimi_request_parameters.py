"""K3 request compatibility; preserve the controller's requested tool restriction."""


def normalize_kimi_parameters(model, parameters):
    result = dict(parameters)
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
