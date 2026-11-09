"""Model views preserve numeric evidence, unknowns, identities and multimodal input."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from mcp_servers.scene_editing.core.adversarial import model_context as views
from scripts.audit_adversarial_context import replay_messages

CORE = Path(__file__).resolve().parents[1] / 'mcp_servers/scene_editing/core/adversarial'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def evidence():
    samples = [dict(time_s=3.000015+i*.500013, track='target', x=12.123456789012345+i,
                    unknown=None, checked=False, road_margin_m=.10000000001) for i in range(24)]
    candidate = dict(candidate_id='candidate_001', failed_checks=['road_surface'],
                     task_failed_checks=['native_replay.keyframe_2'],
                     keyframes=[dict(passed=False, requested=dict(speed_mps=[0, 5]),
                                     actual_ranges=dict(speed_mps=[4.7, 5.21]), limit=.00002)],
                     timeline=samples, scope='Recorded observations; no continuous physical guarantee. ' * 8)
    return dict(instruction='保持前三秒；近险或接触。',
                constraints=dict(task_contract=dict(observation_end_s=3., required_motion='merge_then_align')),
                acceptance_policy=dict(min_margin_m=.1, minimum_progress_fraction=None),
                visual_submission_state=dict(submittable_candidate_ids=['candidate_001']),
                other_actor_snapshots=[dict(time_s=3.000015, actors=samples)],
                last_tool_result=dict(status='no_feasible_goal', candidates=[candidate, deepcopy(candidate)]),
                pdm_self_check=dict(latest_feedback=candidate, candidates=dict(candidate_001=candidate)))


def test_round_trip_retains_all_samples_digits_failures_and_protected_contracts():
    original = evidence(); before = deepcopy(original)
    compact = views.compact_model_context(original)
    assert canonical(views.expand_model_context(compact)) == canonical(original)
    assert original == before
    assert len(canonical(compact)) < len(canonical(original)) * .8
    for name in ('instruction', 'constraints', 'acceptance_policy', 'visual_submission_state'):
        assert compact[name] == original[name]
    assert views.compact_model_context(compact) == compact


def test_nested_object_selection_wrapper_round_trips():
    original = dict(context=evidence(), last_tool_result=None)
    compact = views.compact_model_context(original)
    assert canonical(views.expand_model_context(compact)) == canonical(original)


def test_missing_null_false_integer_float_and_negative_zero_are_distinct():
    rows = [dict(track='same', number=n, note='scope' * 80) for n in [False, 0, 0., -0., 1, 1., True]]
    rows.extend([dict(track='missing', note='scope' * 80), dict(track='null', number=None, note='scope' * 80)])
    original = dict(candidates=rows)
    packed = views.compact_model_context(original)
    assert canonical(views.expand_model_context(packed)) == canonical(original)


def test_nested_ego_catalog_keeps_every_candidate_and_column_order():
    original = dict(candidate_catalog=dict(columns=['candidate_id', 'failed_checks', 'curve_metrics', 'status'], rows=[
        [f'constructor_{i:03d}', ['road_surface'] if i%2 else [],
         dict(start_time_s=3.000015+i/10, max_acceleration_mps2=4.690473212510189,
              scope='These measured curve facts do not certify physical safety.' * 5), 'unknown']
        for i in range(24)]))
    packed = views.compact_model_context(original)
    assert canonical(views.expand_model_context(packed)) == canonical(original)
    assert len(canonical(packed)) < len(canonical(original)) * .8


def test_shared_rows_inside_two_trajectory_tables_decode_before_iteration():
    samples = [dict(time_s=3.000015+i, x_m=1.23456789012345+i, y_m=-.12345,
                    reference_point='rear_axle', scope='sampled reference only') for i in range(24)]
    original = dict(pdm_self_check=dict(a=dict(trajectory=samples), b=dict(trajectory=samples)),
                    last_tool_result=dict(trajectory=samples))
    packed = views.compact_model_context(original)
    assert canonical(views.expand_model_context(packed)) == canonical(original)


@pytest.mark.parametrize('context', [dict(last_tool_result=None), dict(instruction='short'),
    dict(context_ref='user-owned', last_tool_result={'large': 'x'*1000}),
    dict(context_encoding='user-owned', candidates=[1, 2, 3])])
def test_small_inputs_and_reserved_user_keys_are_unchanged(context):
    assert views.compact_model_context(context) == context
    assert views.expand_model_context(context) == context


def test_request_replay_preserves_images_labels_and_surrounding_prompt():
    context = evidence()
    prompt = 'Current geometry-informed state:\n' + json.dumps(context) + '\nRequested diagram follows.'
    messages = [dict(role='user', content=[dict(type='text', text=prompt),
                dict(type='image_url', image_url=dict(url='data:image/png;base64,AAAA')),
                dict(type='text', text='Image 1 is the original road.')])]
    before = deepcopy(messages)
    replay, changes = replay_messages(messages)
    assert messages == before
    assert replay[0]['content'][1:] == messages[0]['content'][1:]
    assert replay[0]['content'][0]['text'].endswith('\nRequested diagram follows.')
    assert changes[0]['after_characters'] < changes[0]['before_characters']
    metrics = views.input_metrics(replay, [dict(function=dict(name='inspect'))])
    assert metrics['image_count'] == 1 and metrics['prompt_tokens'] is None
    assert metrics['text_characters'] < views.input_metrics(messages)['text_characters']
    assert 'AAAA' not in json.dumps(metrics)


def streaming_archive(monkeypatch, tmp_path):
    runtime = types.ModuleType('experiment_runtime')
    class Archive:
        def __init__(self):
            self.folder = tmp_path
            self.profile = dict(api_key='fixture-secret', base_url='https://example.invalid/v1', model='test')
            self.client = None
    runtime.RequestArchive = Archive
    runtime.atomic_json = lambda path, value: Path(path).write_text(json.dumps(value))
    runtime.redact = lambda value, secrets: value
    runtime.utc = lambda: '2026-11-08T00:00:00Z'
    parameters = types.ModuleType('kimi_request_parameters')
    parameters.normalize_kimi_parameters = lambda model, params: params
    monkeypatch.setitem(sys.modules, 'experiment_runtime', runtime)
    monkeypatch.setitem(sys.modules, 'kimi_request_parameters', parameters)
    monkeypatch.setitem(sys.modules, 'model_context', views)
    spec = importlib.util.spec_from_file_location('context_streaming_test', CORE/'streaming_requests.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_real_archive_records_sizes_and_provider_usage_without_credentials(monkeypatch, tmp_path):
    module = streaming_archive(monkeypatch, tmp_path)
    usage = types.SimpleNamespace(model_dump=lambda: dict(prompt_tokens=321, completion_tokens=2))
    chunk = types.SimpleNamespace(model='test', id='fixture', usage=usage, choices=[
        types.SimpleNamespace(delta=types.SimpleNamespace(content='{}', tool_calls=None), finish_reason='stop')])
    monkeypatch.setattr(module, 'deadline_events', lambda *args: iter([('headers', None), ('chunk', chunk)]))
    text, path = module.StreamingRequestArchive().complete('test', [dict(role='user', content='hello 场景')])
    metrics = json.loads((path/'input-metrics.json').read_text())
    assert metrics['prompt_tokens'] == 321 and metrics['token_count_source'] == 'provider_usage'
    assert metrics['text_characters'] == len('hello 场景') and metrics['text_utf8_bytes'] == len('hello 场景'.encode())
    assert 'fixture-secret' not in (path/'input-metrics.json').read_text()


def test_failed_archive_retains_input_measurement_without_fake_token_count(monkeypatch, tmp_path):
    module = streaming_archive(monkeypatch, tmp_path)
    def events(*args):
        yield 'headers', None
        raise module.RequestDeadlineExceeded('fixture')
    monkeypatch.setattr(module, 'deadline_events', events)
    with pytest.raises(RuntimeError, match='Streaming request failed'):
        module.StreamingRequestArchive().complete('failed', [dict(role='user', content='hello')])
    path = next(tmp_path.glob('failed_*/attempt_001'))
    metrics = json.loads((path/'input-metrics.json').read_text())
    assert metrics['text_characters'] == 5 and metrics['prompt_tokens'] is None
    assert json.loads((path/'error.json').read_text())['failure_kind'] == 'request_wall_time_exceeded'


def test_already_compressed_request_can_be_audited_again():
    compact = views.compact_model_context(evidence())
    messages = [dict(role='user', content='Scene state:\n'+json.dumps(compact, ensure_ascii=False, separators=(',', ':')))]
    replay, changes = replay_messages(messages)
    assert replay == messages and changes[0]['already_encoded']


def test_tables_nested_inside_a_list_keep_their_decoding_boundary():
    table = dict(columns=['x_m', 'scope'], rows=[[i+.123456789, 'reference only'*40] for i in range(20)])
    original = dict(last_tool_result=dict(reports=[table, deepcopy(table)]))
    compact = views.compact_model_context(original)
    assert canonical(views.expand_model_context(compact)) == canonical(original)
