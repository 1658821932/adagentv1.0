"""Offline regressions for tool03 defaults, provenance and rerank cleanup."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tools.registry import call_tool
from tools.tool03_video_retriever.schemas import Input, Output


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT', str(tmp_path / 'services'))
    monkeypatch.setenv('SENTRY_RUNTIME_ROOT', str(tmp_path / 'native'))

    def blocked(*args, **kwargs):
        raise AssertionError('offline regression: unexpected network, model or subprocess')

    monkeypatch.setattr('agent.llm_client.LLMClient', blocked)
    monkeypatch.setattr('subprocess.Popen', blocked)
    if importlib.util.find_spec('requests') is not None:
        monkeypatch.setattr('requests.sessions.Session.request', blocked)


@pytest.fixture
def real_result(tmp_path):
    return dict(
        video_path=str(tmp_path / 'clip.mp4'), source_file=str(tmp_path / 'source.mp4'),
        mode='local_hit_reranked', match_score=.9,
        scene_ref=dict(scene_id='scene', scene_token='token', sample_root=str(tmp_path),
                       dataset='fixture', matched_camera='CAM_F0', frame_count=2, fps=2.,
                       manifest_path=str(tmp_path / 'binding.json'),
                       frames_path=str(tmp_path / 'frames.json'),
                       cameras_path=str(tmp_path / 'cameras.json'),
                       cameras={}, images_complete=False))


@pytest.mark.parametrize('timeout', [None, 37])
def test_public_real_entry_honors_default_and_explicit_timeout(monkeypatch, real_result, timeout):
    from mcp_servers.sentry_video_search.adapter import client

    calls = []

    def completed(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(
            dict(status='success', result=deepcopy(real_result))))

    monkeypatch.setattr(client, 'VIDEO_RETRIEVER_PYTHON', Path(sys.executable))
    monkeypatch.setattr(client, 'run_logged', completed)
    arguments = dict(requirement='a road', candidate_id='cand_2')
    if timeout is not None:
        arguments['timeout'] = timeout
    result = call_tool('video_retriever', arguments, mode='real')
    assert result['status'] == 'success', result
    assert result['result']['candidate_id'] == 'cand_2'
    assert calls[0]['timeout'] == (1800 if timeout is None else timeout)
    assert Input.model_fields['timeout'].default == 1800


def test_mock_preserves_candidate_id_and_remains_unbound():
    result = call_tool('video_retriever', dict(requirement='a road', candidate_id='cand_2'), mode='mock')
    assert result['status'] == 'success', result
    assert result['result']['candidate_id'] == 'cand_2'
    assert result['result']['scene_ref'] is None
    Output.model_validate(result['result'])


def test_verbose_scenario_text_is_rejected_as_a_retrieval_query():
    with pytest.raises(ValidationError):
        Input.model_validate(dict(requirement='视频需包含白天多云的丁字路口、自车右转且向左偏离本车道的接近过程，以及右侧支路一辆保持本车道直行且可调整速度与到达时机的车辆。'))


def test_output_accepts_bound_rank_two_and_three_alternatives(real_result, tmp_path):
    alternative = dict(
        rank=2, video_path=str(tmp_path / 'second.mp4'), match_score=.7,
        source_file=str(tmp_path / 'second-source.mp4'),
        source_segment={'start_time': 1., 'end_time': 2.},
        scene_ref=dict(real_result['scene_ref'], scene_id='scene-2', scene_token='token-2'))
    output = Output.model_validate(dict(real_result, alternatives=[alternative]))
    assert output.alternatives[0].rank == 2
    assert output.alternatives[0].scene_ref.scene_token == 'token-2'


def test_compact_history_preserves_candidate_id(real_result, tmp_path):
    from agent.context import compact_state

    result = dict(real_result, candidate_id='cand_2')
    state = dict(history=[dict(tool='video_retriever', result=dict(
        status='success', tool='video_retriever', result=result),
        result_ref=str(tmp_path / 'saved-result.json'))])
    compact = compact_state(state)
    assert compact['history'][0]['result']['result']['candidate_id'] == 'cand_2'
    assert compact['history'][0]['result']['result_ref'] == str(tmp_path / 'saved-result.json')


@pytest.mark.parametrize('score', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_scores_are_rejected(real_result, score):
    with pytest.raises(ValidationError):
        Output.model_validate(dict(real_result, match_score=score))


@pytest.mark.parametrize('missing', ['source_file', 'scene_ref'])
def test_non_mock_success_requires_scene_provenance(real_result, missing):
    result = dict(real_result)
    result.pop(missing)
    with pytest.raises(ValidationError):
        Output.model_validate(result)


def test_finite_scores_are_not_assumed_to_be_probabilities(real_result):
    assert Output.model_validate(dict(real_result, match_score=-.1)).match_score == -.1


def test_public_wrapper_rejects_unbound_real_success(monkeypatch, real_result):
    from tools.tool03_video_retriever import real

    monkeypatch.setattr(real, 'retrieve_video', lambda data: dict(real_result, scene_ref=None))
    result = call_tool('video_retriever', dict(requirement='a road'), mode='real')
    assert result['status'] == 'error'
    assert result['error']['type'] == 'output_validation_error'


@pytest.fixture
def fake_rerank(monkeypatch):
    pytest.importorskip('requests')
    from mcp_servers.sentry_video_search.core import reranker

    profile = dict(api_key='offline-fixture', rerank_model='offline',
                   rerank_url='https://example.invalid', rerank_timeout=1)
    monkeypatch.setattr('mcp_servers.sentry_video_search.core.api_settings.get_profile',
                        lambda *a, **kw: profile)
    monkeypatch.setattr(reranker, '_settings', lambda: ('a', 'b', 'c', 'd', 'p', ''))

    def trim(*args, **kwargs):
        Path(kwargs.get('output_path') or args[3]).write_bytes(b'x' * 25000)

    monkeypatch.setattr(reranker, 'trim_clip', trim)
    monkeypatch.setattr(reranker, 'preprocess_chunk', lambda path, **kw: path)
    return reranker


def hits():
    return [dict(source_file=f'/{i}.mp4', start_time=0., end_time=1., similarity_score=.8)
            for i in range(2)]


def test_failed_first_upload_cleans_a_later_success(monkeypatch, tmp_path, fake_rerank):
    uploaded, deleted = [], []
    later_uploaded = Event()

    def upload(path, key, settings):
        if path.stem == 'candidate_00':
            assert later_uploaded.wait(5), 'second worker did not complete'
            raise RuntimeError('fixture upload failure')
        url = 'https://example.invalid/staged/1.mp4'
        uploaded.append(url)
        later_uploaded.set()
        return url

    monkeypatch.setattr(fake_rerank, '_upload', upload)
    monkeypatch.setattr(fake_rerank, '_delete', lambda url, settings: deleted.append(url))
    with pytest.raises(RuntimeError, match='fixture upload failure'):
        fake_rerank.rerank_hits('a road', hits(), run_dir=tmp_path / 'run')
    assert uploaded == deleted == ['https://example.invalid/staged/1.mp4']


def test_success_keeps_document_indices_and_cleans_all_uploads(monkeypatch, tmp_path, fake_rerank):
    deleted, captured = [], []

    def upload(path, key, settings):
        return f'https://example.invalid/staged/{path.stem}.mp4'

    def audited(call, payload, **kwargs):
        captured.append(payload)
        return SimpleNamespace(status_code=200), dict(output=dict(results=[
            dict(index=1, relevance_score=.9), dict(index=0, relevance_score=.3)]))

    monkeypatch.setattr(fake_rerank, '_upload', upload)
    monkeypatch.setattr(fake_rerank, '_delete', lambda url, settings: deleted.append(url))
    monkeypatch.setattr('common.llm_audit.audited_json_call', audited)
    winner, details = fake_rerank.rerank_hits('a road', hits(), run_dir=tmp_path / 'run')
    assert winner['source_file'] == '/1.mp4'
    documents = captured[0]['input']['documents']
    assert documents == [dict(video=f'https://example.invalid/staged/candidate_{i:02d}.mp4')
                         for i in range(2)]
    assert sorted(deleted) == sorted(document['video'] for document in documents)
    assert details['ranked_results'][0]['candidate_index'] == 1
