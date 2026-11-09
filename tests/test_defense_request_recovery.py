"""Transient transport retry preserves stage inputs and cannot repeat indefinitely."""
import json
from pathlib import Path

import pytest
from mcp_servers.scene_editing.core.adversarial.defense_request_recovery import retryable_request,run_with_one_retry


def failed(run,number=1,error_type='APIConnectionError',chunks=0):
    run.mkdir(exist_ok=True)
    (run/'summary.json').write_text(json.dumps(dict(stop_reason='request_or_execution_failed',selected_candidate_id=None,model_actions=0)))
    attempt=run/f'requests/turn_01_{number}/attempt_001';attempt.mkdir(parents=True)
    (attempt/'error.json').write_text(json.dumps(dict(error_type=error_type,chunks_received=chunks,failure_phase='waiting_response_headers')))


def test_retry_passes_same_arguments_and_only_sets_resume(tmp_path):
    calls=[];options=dict(timeout_seconds=300,max_tokens=16384)
    def editor(*args,**kwargs):
        calls.append((args,kwargs))
        if len(calls)==1:
            failed(tmp_path/'run');raise RuntimeError('connection')
        return dict(status='submitted')
    result=run_with_one_retry(editor,('scene','constructor'),options,tmp_path/'run')
    assert result['status']=='submitted' and calls[0][0]==calls[1][0]
    assert calls[1][1]==dict(options,resume=True)
    assert options==dict(timeout_seconds=300,max_tokens=16384)


def test_two_failed_attempts_are_not_retried_again(tmp_path):
    failed(tmp_path/'run');failed(tmp_path/'run',2)
    assert retryable_request(tmp_path/'run') is None


@pytest.mark.parametrize('kind,chunks',[('BadRequestError',0),('APIConnectionError',1)])
def test_parameter_or_partial_output_failure_not_replayed(tmp_path,kind,chunks):
    failed(tmp_path/'run',error_type=kind,chunks=chunks)
    assert retryable_request(tmp_path/'run') is None


def test_new_retry_failure_propagates(tmp_path):
    calls=[]
    def editor(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==1:failed(tmp_path/'run')
        else:failed(tmp_path/'run',2)
        raise RuntimeError('still disconnected')
    with pytest.raises(RuntimeError,match='still disconnected'):
        run_with_one_retry(editor,(),{},tmp_path/'run')
    assert len(calls)==2
