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
    request_dirs = sorted((run_dir/'requests').glob(f'turn_{turn:02d}_*'))
    if len(request_dirs) != 1:
        return None
    attempts = sorted(request_dirs[0].glob('attempt_*'))
    if not attempts or any((item/'response.json').exists() for item in attempts):
        return None
    latest = attempts[-1]
    path = latest/'error.json'
    if not path.is_file():
        return None
    error = json.loads(path.read_text())
    retryable_types = {'APIConnectionError', 'APITimeoutError', 'NoOutputTimeout', 'RequestDeadlineExceeded'}
    zero_output = error.get('chunks_received') == 0 or error.get('output_received') is False
    if error.get('error_type') not in retryable_types or not zero_output:
        return None
    return dict(turn=turn, failed_attempt=str(latest), error_type=error['error_type'])


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
