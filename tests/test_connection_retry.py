"""Main-model transport retries are bounded, observable and never replay partial streams."""
import pytest
from openai import APIConnectionError
from agent.connection_retry import call_with_connection_retries
from common.llm_audit import trace


def disconnected():
    return APIConnectionError(message='fixture disconnect', request=object())


def test_three_retries_use_backoff_and_audit_attempt_numbers():
    calls = []
    sleeps = []
    notices = []

    def operation():
        calls.append(trace().get('llm_attempt'))
        if len(calls) <= 3:
            raise disconnected()
        return 'ok'

    assert call_with_connection_retries(operation, sleep=sleeps.append,
        notify=lambda attempt, delay, error: notices.append((attempt, delay))) == 'ok'
    assert calls == [1, 2, 3, 4]
    assert sleeps == [1.0, 2.0, 4.0]
    assert notices == [(1, 1.0), (2, 2.0), (3, 4.0)]


def test_fourth_connection_failure_is_not_retried_again():
    calls = []
    sleeps = []

    def operation():
        calls.append(1)
        raise disconnected()

    with pytest.raises(APIConnectionError):
        call_with_connection_retries(operation, sleep=sleeps.append)
    assert len(calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_non_connection_error_fails_immediately():
    calls = []

    def operation():
        calls.append(1)
        raise ValueError('invalid request')

    with pytest.raises(ValueError, match='invalid request'):
        call_with_connection_retries(operation, sleep=lambda _: None)
    assert len(calls) == 1


def test_partial_stream_is_not_replayed():
    calls = []
    sleeps = []

    def operation():
        calls.append(1)
        raise disconnected()

    with pytest.raises(APIConnectionError):
        call_with_connection_retries(operation, metrics=lambda: {'chunks': 1}, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []


def test_main_function_stream_retries_before_first_chunk(monkeypatch):
    from types import SimpleNamespace as NS
    from agent.llm_client import LLMClient

    monkeypatch.setattr('time.sleep', lambda _: None)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise disconnected()
        chunk = NS(usage=None, choices=[NS(index=0,
            delta=NS(reasoning_content=None, content=None, tool_calls=[]), finish_reason='stop')])
        class Stream:
            def __iter__(self): return iter([chunk])
            def close(self): return None
        return Stream()

    client = LLMClient.__new__(LLMClient)
    client.model = 'fixture-model'
    client.max_tokens = 128
    client.base_url = 'https://example.invalid'
    client.api_key = 'fixture-only'
    client.request_profile = {}
    client.client = NS(chat=NS(completions=NS(create=create)))
    result = client.call_functions([], [])
    assert result['finish_reason'] == 'stop' and len(calls) == 2
