"""One unchanged recovery for a new defense request with no received stream chunks."""
from pathlib import Path
import json



def retryable_request(run_dir):
    run_dir = Path(run_dir)
    summary_path = run_dir/'summary.json'
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get('stop_reason') != 'request_or_execution_failed' or summary.get('selected_candidate_id') is not None:
        return None
    turn = summary['model_actions']+1
    attempts = list((run_dir/'requests').glob(f'turn_{turn:02d}_*/attempt_001'))
    if len(attempts) != 1 or (attempts[0]/'response.json').exists():
        return None
    path = attempts[0]/'error.json'
    if not path.is_file():
        return None
    error = json.loads(path.read_text())
    if (error.get('error_type') not in {'APIConnectionError', 'APITimeoutError'}
            or error.get('chunks_received') != 0 or error.get('failure_phase') != 'waiting_response_headers'):
        return None
    return dict(turn=turn, failed_attempt=str(attempts[0]), error_type=error['error_type'])


def run_with_one_retry(function, args, options, run_dir):
    try:
        return function(*args, **options)
    except Exception:
        request = retryable_request(run_dir)
        if request is None:
            raise
        path = Path(run_dir)/'status-history'/f"automatic-recovery-turn-{request['turn']:02d}.json"
        if path.exists():
            raise
        path.parent.mkdir(exist_ok=True)
        with path.open('x') as stream:
            json.dump(dict(request, policy='one unchanged retry; editor verifies exact request payload'), stream, indent=2)
        print('Defense request connection failed before headers; resuming once with unchanged configuration', flush=True)
        return function(*args, **dict(options, resume=True))
