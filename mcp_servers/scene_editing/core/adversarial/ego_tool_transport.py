"""Require native results, with explicit image transport capabilities and correction records."""
from copy import deepcopy
import json
from pathlib import Path


class NativeToolProtocolError(RuntimeError):
    """A completed response failed the required native-call contract."""


def native_wire_choice(messages, image_tool_choice='required', tool_choice_policy='preserve'):
    if tool_choice_policy not in ('preserve','auto'):
        raise ValueError('tool_choice_policy must be preserve or auto')
    if tool_choice_policy == 'auto':
        return 'auto'
    if image_tool_choice not in ('auto', 'required'):
        raise ValueError('image_tool_choice must be auto or required')
    has_image = any(isinstance(message.get('content'), list) and any(
        isinstance(part, dict) and part.get('type') == 'image_url'
        for part in message['content']) for message in messages or [])
    return image_tool_choice if has_image else 'required'


def require_native_call(choice, tools, *, messages=None, image_tool_choice='required', tool_choice_policy='preserve'):
    if not tools:
        raise ValueError('A native-call phase must expose at least one function')
    if not isinstance(choice, dict):
        if choice not in ('auto', 'required'):
            raise ValueError('Native-call phases cannot disable tool calls')
        return native_wire_choice(messages, image_tool_choice, tool_choice_policy), tools
    name = choice['function']['name']
    selected = [deepcopy(tool) for tool in tools if tool['function']['name'] == name]
    if len(selected) != 1:
        raise ValueError('Forced phase function is missing or ambiguous: '+name)
    return native_wire_choice(messages, image_tool_choice, tool_choice_policy), selected


def protocol_rejection(run_dir, attempt, turn, calls):
    """No executable call is invented, and no model text is interpreted as code."""
    if len(calls) == 1:
        raise ValueError('A single native call is not a missing-call rejection')
    feedback = dict(status='invalid', error_code='native_tool_call_required', executed=False,
        received_call_count=len(calls),
        error='No tool was executed. Return exactly one native function call from the currently exposed tools. '
              'Do not put a command in prose or a JSON code block. Use finish_search only as an explicit native call if you intend to end the search.')
    return dict(schema_version='native_tool_protocol.v1', turn=turn,
        request_path=str(Path(attempt).relative_to(run_dir)), feedback=feedback)


def restore_protocol_feedback(path, run_dir, profile):
    """Verify a completed, non-executed response before restoring its budget use."""
    run_dir = Path(run_dir).resolve()
    record = json.loads(Path(path).read_text())
    attempt = (run_dir / record['request_path']).resolve()
    if not attempt.is_relative_to(run_dir / 'requests') or (attempt / 'error.json').exists():
        raise ValueError('Protocol record must reference a completed local request')
    request = json.loads((attempt / 'request.json').read_text())
    response = json.loads((attempt / 'response.json').read_text())
    result = json.loads((attempt / 'result.json').read_text())
    calls = response['choices'][0]['message'].get('tool_calls') or []
    feedback = record['feedback']
    if (record.get('schema_version') != 'native_tool_protocol.v1'
            or request.get('endpoint') != profile['base_url']
            or request['payload'].get('model') != profile['model']
            or request['payload'].get('tool_choice') != native_wire_choice(
                request['payload'].get('messages'), profile.get('image_tool_choice', 'required'), profile.get('tool_choice_policy','preserve'))
            or result.get('status') != 'received' or len(calls) == 1
            or feedback != protocol_rejection(run_dir, attempt, record['turn'], calls)['feedback']):
        raise ValueError('Archived protocol rejection or request provenance differs')
    return feedback


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
