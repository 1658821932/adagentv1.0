"""Lossless model-only views; raw reports, policies and numeric precision stay intact."""
from collections import Counter
import json


ENCODING = 'lossless_context.v1'
# Only measured scene evidence is encoded. Instructions, policies and tool schemas
# retain their normal representation, including inside an object-selection wrapper.
EVIDENCE_FIELDS = frozenset((
    'timeline', 'ego_timeline', 'recorded_scene_timeline', 'candidates',
    'candidate_catalog', 'other_actor_snapshots', 'actors_at_start',
    'existing_vehicles', 'vehicle_planning_hints', 'road_sections',
    'last_tool_result', 'pdm_self_check', 'active_candidate_details',
    'edit_history', 'actual_start_facts'))
NOTE = ('Expand context_ref from shared_evidence. For record_table=v1, each row '
        'inherits shared fields plus columns zipped to row values. For column_table=v1, '
        'columns fixes the original column order; shared holds constant columns and '
        'data holds the other column arrays (which may themselves be record tables). '
        'Every sample, field, unknown/null/false value and numeric digit is retained. '
        'References are exact evidence reuse, not new measurements.')


def _json(value, canonical=False):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=canonical)


def _smaller(original, packed):
    return packed if len(_json(packed)) + 80 < len(_json(original)) else original


def _tables(value):
    if isinstance(value, dict):
        # Existing ego catalogs already use row tables. Transpose their nested
        # object-valued columns so their field names need not repeat per candidate.
        if (set(value) == {'columns', 'rows'} and isinstance(value['columns'], list)
                and all(isinstance(k, str) for k in value['columns'])
                and len(set(value['columns'])) == len(value['columns'])
                and isinstance(value['rows'], list) and len(value['rows']) >= 2
                and all(isinstance(row, list) and len(row) == len(value['columns']) for row in value['rows'])):
            columns = {key: [row[i] for row in value['rows']] for i, key in enumerate(value['columns'])}
            shared = {key: items[0] for key, items in columns.items() if all(_json(item, canonical=True) == _json(items[0], canonical=True) for item in items[1:])}
            packed = dict(column_table='v1', columns=value['columns'], row_count=len(value['rows']),
                          shared={key: _tables(item) for key, item in shared.items()},
                          data={key: _tables(items) for key, items in columns.items() if key not in shared})
            return _smaller(value, packed)
        return {key: _tables(item) for key, item in value.items()}
    if not isinstance(value, list):
        return value
    items = [_tables(item) for item in value]
    if len(items) < 2 or not all(isinstance(item, dict) for item in value):
        return items
    if any(set(packed) != set(raw) for packed, raw in zip(items, value)):
        return items  # A nested table must retain its own decoding boundary.
    keys = list(value[0])
    if not all(set(item) == set(keys) for item in value):
        return items  # Missing and explicit null must never be conflated.
    shared = {key: items[0][key] for key in keys if all(_json(item[key], canonical=True) == _json(items[0][key], canonical=True) for item in items[1:])}
    columns = [key for key in keys if key not in shared]
    packed = dict(record_table='v1', shared=shared, columns=columns,
                  rows=[[item[key] for key in columns] for item in items])
    return _smaller(items, packed)


def _reserved(value):
    if isinstance(value, dict):
        return bool({'context_ref', 'context_encoding', 'shared_evidence', 'record_table', 'column_table'} & value.keys()) or any(_reserved(v) for v in value.values())
    return isinstance(value, list) and any(_reserved(v) for v in value)


def compact_model_context(context):
    """Build a deterministic view of selected evidence without mutating its source.

    Reject no user data: a reserved-key collision or insufficient savings simply
    leaves the view unchanged. Running twice is harmless. No rounding or truncation.
    """
    if _reserved(context):
        return context
    result = dict(context)
    if isinstance(context.get('context'), dict):
        result['context'] = compact_model_context(context['context'])
    evidence = {key: _tables(value) for key, value in result.items() if key in EVIDENCE_FIELDS}
    counts = Counter()

    def identity(value):
        if isinstance(value, (dict, list, str)):
            text = _json(value, canonical=True)
            if len(text) >= 160:
                return text
        return None

    def count(value):
        key = identity(value)
        if key is not None:
            counts[key] += 1
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        for child in children:
            count(child)

    for value in evidence.values():
        count(value)
    shared, names = {}, {}

    def encode(value):
        def descend():
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [encode(item) for item in value]
            return value
        key = identity(value)
        if key is not None and counts[key] > 1:
            if key not in names:
                name = 'e' + str(len(names))
                names[key] = name
                shared[name] = descend()
            return dict(context_ref=names[key])
        return descend()

    result.update({key: encode(value) for key, value in evidence.items()})
    if any(_json(result[key]) != _json(context[key]) for key in evidence):
        result['shared_evidence'] = shared
        result['context_encoding'] = dict(version=ENCODING, instructions=NOTE)
    # Small stages should not pay a metadata overhead for negligible savings.
    return result if len(_json(result)) + 128 < len(_json(context)) else context


def expand_model_context(context):
    """Reconstruct the original JSON value for offline audits and regression checks."""
    metadata = context.get('context_encoding')
    encoded = isinstance(metadata, dict) and metadata.get('version') == ENCODING and 'shared_evidence' in context
    shared = context.get('shared_evidence', {}) if encoded else {}

    def expand(value):
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {'context_ref'} and encoded:
            return expand(shared[value['context_ref']])
        if value.get('record_table') == 'v1' and encoded:
            common = expand(value['shared'])
            columns, rows = expand(value['columns']), expand(value['rows'])
            return [dict(common, **dict(zip(columns, row))) for row in rows]
        if value.get('column_table') == 'v1' and encoded:
            common, data = expand(value['shared']), expand(value['data'])
            columns = expand(value['columns'])
            return dict(columns=columns, rows=[
                [common[key] if key in common else data[key][i] for key in columns]
                for i in range(value['row_count'])])
        if isinstance(value.get('context_encoding'), dict) and value['context_encoding'].get('version') == ENCODING:
            return expand_model_context(value)
        return {key: expand(item) for key, item in value.items()}

    return {key: expand(value) for key, value in context.items()
            if not encoded or key not in ('shared_evidence', 'context_encoding')}


def input_metrics(messages, tools=()):
    """Exact text/schema sizes; image bytes are not text tokens or context tokens."""
    texts, images = [], 0
    for message in messages:
        content = message.get('content')
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get('type') == 'text':
                    texts.append(part['text'])
                elif part.get('type') == 'image_url':
                    images += 1
    schema = _json(tools) if tools else ''
    return dict(message_count=len(messages), image_count=images,
                text_characters=sum(map(len, texts)), text_utf8_bytes=sum(len(text.encode()) for text in texts),
                tool_schema_characters=len(schema), tool_schema_utf8_bytes=len(schema.encode()),
                prompt_tokens=None, token_count_source='unavailable_until_provider_usage')
