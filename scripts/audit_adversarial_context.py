"""Audit archived model inputs and replay lossless context compression without API calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp_servers.scene_editing.core.adversarial.model_context import (
    compact_model_context, expand_model_context, input_metrics)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def context_span(text):
    markers = ('Current geometry-informed state:\n', 'Scene state:\n',
               'State:\n', '{"context":', '{"instruction":')
    positions = sorted({text.find(marker) + (0 if marker.startswith('{') else len(marker))
                        for marker in markers if text.find(marker) >= 0})
    for start in positions:
        try:
            context, length = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(context, dict):
            return start, start + length, context
    return None


def replay_messages(messages):
    """Keep images and surrounding prompt text byte-for-byte; replace context only."""
    result = deepcopy(messages)
    changes = []
    for message in result:
        content = message.get('content')
        parts = [dict(text=content)] if isinstance(content, str) else [
            part for part in content or [] if part.get('type') == 'text']
        for part in parts:
            span = context_span(part['text'])
            if span is None:
                continue
            start, end, original = span
            compressed = compact_model_context(original)
            if dump(expand_model_context(compressed)) != dump(expand_model_context(original)):
                raise ValueError('Lossless context round-trip failed')
            part['text'] = part['text'][:start] + json.dumps(compressed, ensure_ascii=False, separators=(',', ':')) + part['text'][end:]
            changes.append(dict(already_encoded=dump(expand_model_context(original)) != dump(original), before_characters=len(dump(original)), after_characters=len(dump(compressed)),
                                field_characters={key: len(dump(value)) for key, value in original.items()}))
        if isinstance(content, str):
            message['content'] = parts[0]['text']
    return result, changes


def _row(source, stage, messages, tools, usage=None, reconstructed=False):
    replay, changes = replay_messages(messages)
    before, after = input_metrics(messages, tools), input_metrics(replay, tools)
    before.update(prompt_tokens=(usage or {}).get('prompt_tokens'),
                  token_count_source='provider_usage' if (usage or {}).get('prompt_tokens') is not None else 'unavailable')
    return make_row(source, stage, before, after, changes, reconstructed)


def make_row(source, stage, before, after, changes, reconstructed=False):
    a = before['text_characters'] + before['tool_schema_characters']
    b = after['text_characters'] + after['tool_schema_characters']
    return dict(source=source, stage=stage, reconstructed=reconstructed,
                before=before, after=after, text_and_tools_characters_before=a,
                text_and_tools_characters_after=b, reduction_percent=round(100*(1-b/a), 2) if a else 0,
                contexts=changes)


def audit_request(path):
    request = json.loads(path.read_text())
    payload = request.get('payload') or {}
    if not isinstance(payload.get('messages'), list):
        return None
    name = path.parent.parent.name
    stage = ('ego_understanding' if 'defense_understanding' in path.parts else
             'ego_editing' if 'defense_agent' in path.parts else
             'object_selection' if name.startswith('object_') else
             'attack_editing' if name.startswith('edit_') else 'other_llm')
    usage = None
    for filename in ('response.json', 'partial-response.json'):
        response = path.with_name(filename)
        if response.is_file():
            usage = json.loads(response.read_text()).get('usage')
            if usage:
                break
    return _row(str(path), stage, payload['messages'], payload.get('tools', ()), usage)




def audit(roots):
    rows = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError('Audit root must be an existing directory: ' + str(root))
        for path in sorted(root.rglob('request.json')):
            if path.parent.name != 'attempt_001' or path.resolve() in seen:
                continue
            seen.add(path.resolve())
            row = audit_request(path)
            if row:
                rows.append(row)
    stages = {}
    for stage in sorted({row['stage'] for row in rows}):
        group = [row for row in rows if row['stage'] == stage]
        before = sum(row['text_and_tools_characters_before'] for row in group)
        after = sum(row['text_and_tools_characters_after'] for row in group)
        tokens = [row['before']['prompt_tokens'] for row in group if row['before']['prompt_tokens'] is not None]
        stages[stage] = dict(requests=len(group), total_characters_before=before, total_characters_after=after,
                             reduction_percent=round(100*(1-after/before), 2) if before else 0,
                             measured_prompt_token_range=[min(tokens), max(tokens)] if tokens else None)
    return dict(schema_version='context_audit.v1', api_requests_made=0, all_contexts_round_trip=True,
                measurement_scope='Exact text and tool-schema sizes. Provider prompt_tokens include images when present. After tokens are unmeasured. Video text is reconstructed; image token cost is unknown.',
                stages=stages, requests=rows)


def markdown(report):
    lines = ['# LLM context audit', '', report['measurement_scope'], '',
             '| Stage | Requests | Text + schemas before | After | Reduction | Original input tokens |',
             '|---|---:|---:|---:|---:|---|']
    for stage, row in report['stages'].items():
        tokens = row['measured_prompt_token_range']
        lines.append(f"| {stage} | {row['requests']} | {row['total_characters_before']:,} | {row['total_characters_after']:,} | {row['reduction_percent']}% | {str(tokens) if tokens else 'unavailable'} |")
    lines += ['', 'All compressed contexts round-trip exactly. No API calls; original artifacts remain untouched.', '',
              'Constructor, independent checks, PDM and rendering have no model context in these runs. The top-level orchestrator was not exercised by these archived driver runs.', '',
              '| Request | Before characters | After | Reduction |', '|---|---:|---:|---:|']
    for row in report['requests']:
        lines.append(f"| {row['source']} | {row['text_and_tools_characters_before']:,} | {row['text_and_tools_characters_after']:,} | {row['reduction_percent']}% |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, help='New JSON report; existing reports are never overwritten')
    args = parser.parse_args()
    report = audit(args.roots)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        md_path = args.output.with_suffix('.md')
        if args.output.exists() or md_path.exists():
            raise FileExistsError('Choose a new audit output')
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        with md_path.open('x') as stream:
            stream.write(markdown(report))
    print(json.dumps(report['stages'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
