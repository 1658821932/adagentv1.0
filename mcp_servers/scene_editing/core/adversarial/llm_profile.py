"""Read the explicitly injected API profile without importing the host application."""
import os


def load_llm_profile():
    names = dict(api_key='ADVERSARIAL_API_KEY', base_url='ADVERSARIAL_API_BASE_URL', model='ADVERSARIAL_API_MODEL')
    profile = {field: os.environ.get(name, '') for field, name in names.items()}
    missing = [names[field] for field, value in profile.items() if not value.strip()]
    if missing:
        raise RuntimeError('API profile must be supplied by the host adapter; missing: ' + ', '.join(missing))
    for field, name in [('timeout','ADVERSARIAL_API_TIMEOUT'), ('max_tokens','ADVERSARIAL_API_MAX_TOKENS')]:
        if name in os.environ:
            profile[field] = int(os.environ[name])
    if os.environ.get('ADVERSARIAL_IMAGE_TOOL_CHOICE'):
        choice = os.environ['ADVERSARIAL_IMAGE_TOOL_CHOICE']
        if choice not in ('auto','required'):
            raise RuntimeError('ADVERSARIAL_IMAGE_TOOL_CHOICE must be auto or required')
        profile['image_tool_choice'] = choice
    if os.environ.get('ADVERSARIAL_REQUIRED_TOOL_THINKING'):
        mode = os.environ['ADVERSARIAL_REQUIRED_TOOL_THINKING']
        if mode not in ('preserve', 'disabled'):
            raise RuntimeError('ADVERSARIAL_REQUIRED_TOOL_THINKING must be preserve or disabled')
        profile['required_tool_thinking'] = mode
    for field, name in [('request_dialect','ADVERSARIAL_REQUEST_DIALECT'), ('thinking_mode','ADVERSARIAL_THINKING_MODE'), ('tool_choice_policy','ADVERSARIAL_TOOL_CHOICE_POLICY'), ('reasoning_effort','ADVERSARIAL_REASONING_EFFORT')]:
        if os.environ.get(name):
            profile[field] = os.environ[name]
    if profile.get('thinking_mode'):
        from llm_request_policy import validate_request_policy
        validate_request_policy(profile)
    return profile


def reasoning_effort(default='low'):
    """Kimi reasoning effort; host adapter injects ADVERSARIAL_REASONING_EFFORT."""
    effort = os.environ.get('ADVERSARIAL_REASONING_EFFORT', default)
    if effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
        raise RuntimeError('ADVERSARIAL_REASONING_EFFORT must be low, medium, high, xhigh or max')
    return effort
