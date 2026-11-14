"""Candidate-level process parallelism preserves results and enforces local timeouts."""
from functools import partial
import time

from mcp_servers.scene_editing.core.adversarial.foci_parallel import optimize_candidates


def delayed(value, delay):
    time.sleep(delay)
    return value


def fail(message):
    raise ValueError(message)


def test_two_candidates_run_in_parallel():
    started = time.monotonic()
    results = optimize_candidates([
        dict(index=1, optimize=partial(delayed, 'one', 0.35)),
        dict(index=2, optimize=partial(delayed, 'two', 0.35)),
    ], max_workers=2)
    elapsed = time.monotonic()-started
    assert [result['value'] for result in results] == ['one', 'two']
    assert elapsed < 0.65


def test_timeout_does_not_discard_other_candidate():
    results = optimize_candidates([
        dict(index=1, timeout_s=0.08, optimize=partial(delayed, 'slow', 0.4)),
        dict(index=2, timeout_s=1, optimize=partial(delayed, 'fast', 0.02)),
    ], max_workers=2)
    assert results[0]['error_type'] == 'TimeoutExpired'
    assert results[1]['value'] == 'fast'


def test_worker_error_preserves_successful_results():
    results = optimize_candidates([
        dict(index=1, optimize=partial(fail, 'fixture')),
        dict(index=2, optimize=partial(delayed, 'ok', 0.01)),
    ], max_workers=2)
    assert results[0]['error_type'] == 'ValueError'
    assert results[1]['value'] == 'ok'
