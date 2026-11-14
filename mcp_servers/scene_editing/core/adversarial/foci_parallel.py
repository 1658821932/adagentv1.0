"""Bounded candidate-level parallelism with per-worker timeout and partial results."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import signal


class CandidateTimeout(Exception):
    pass


def _execute(task):
    timeout = task.get('timeout_s')
    if timeout is not None:
        def expired(signum, frame):
            raise CandidateTimeout(f"FOCI candidate exceeded {timeout}s")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        return dict(index=task['index'], value=task['optimize']())
    except CandidateTimeout as error:
        return dict(index=task['index'], error_type='TimeoutExpired', error=str(error))
    except Exception as error:
        return dict(index=task['index'], error_type=type(error).__name__, error=str(error))
    finally:
        if timeout is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)


def optimize_candidates(tasks, max_workers=2):
    if not tasks:
        return []
    workers = min(max(1, int(max_workers)), 2, len(tasks))
    if workers == 1 or len(tasks) == 1:
        return sorted((_execute(task) for task in tasks), key=lambda result: result['index'])
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        futures = [pool.submit(_execute, task) for task in tasks]
        results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda result: result['index'])
