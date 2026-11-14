"""Early-stop rules for repeated requests and unproductive candidate searches."""


def failure_signature(name, result):
    if not isinstance(result, dict):
        return None
    status = result.get('status')
    if status == 'invalid':
        return ('invalid', name, str(result.get('error', '')).split(':', 1)[0])
    if status not in {'fallback', 'no_feasible_goal', 'preview', 'not_feasible'}:
        return None
    failures = set(result.get('failed_checks', [])) | set(result.get('task_failed_checks', []))
    if not failures:
        return None
    return (status, name, tuple(sorted(failures)))


def failure_cost(result):
    candidates = result.get('candidates')
    if not isinstance(candidates, list):
        return None
    costs = [candidate.get('guidance_cost') for candidate in candidates
             if isinstance(candidate, dict) and isinstance(candidate.get('guidance_cost'), (int, float))]
    return min(costs) if costs else None


def repetition(previous, name, result):
    signature = failure_signature(name, result)
    cost = failure_cost(result)
    if signature is None:
        return None
    if previous is None or previous[0] != signature:
        return signature, 1, cost
    count = previous[1]
    previous_cost = previous[2]
    if cost is not None and previous_cost is not None and cost < previous_cost * .95:
        return signature, 1, cost
    return signature, count + 1, cost if cost is not None else previous_cost
