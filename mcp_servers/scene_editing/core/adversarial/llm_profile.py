"""Read the explicitly injected API profile without importing the host application."""
import os


def load_llm_profile():
    names = dict(api_key='ADVERSARIAL_API_KEY', base_url='ADVERSARIAL_API_BASE_URL', model='ADVERSARIAL_API_MODEL')
    profile = {field: os.environ.get(name, '') for field, name in names.items()}
    missing = [names[field] for field, value in profile.items() if not value.strip()]
    if missing:
        raise RuntimeError('API profile must be supplied by the host adapter; missing: ' + ', '.join(missing))
    return profile
