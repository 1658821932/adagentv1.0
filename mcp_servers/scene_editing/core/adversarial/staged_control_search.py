"""Search continuations of reference contacts in the driving simulator."""

import numpy as np
from scipy.optimize import least_squares


MAX_PREFIXES = 5


def continuation_priority(item):
    cost, _, record = item
    metrics = record['search_metrics']
    other_gap = metrics['min_other_signed_gap_m']
    return (
        not metrics['geometry_motion_guidance_feasible'],
        not metrics['window_reference_contact'],
        metrics['motion_guidance_violation_count'],
        max(0., -metrics['min_road_signed_margin_m'])
        + max(0., -other_gap if other_gap is not None else 0.),
        cost,
    )


def select_prefixes(records, required_margin, last_knot):
    eligible = []
    for item in records:
        prefix = item[2]['search_metrics'].get('contact_prefix')
        if (prefix is not None and prefix['third_party_clear']
                and prefix['time_s'] <= last_knot
                and prefix['road_margin_m'] >= required_margin - 1e-8
                and not prefix['motion_violations']):
            eligible.append(item)

    chosen = []
    for item in sorted(eligible, key=continuation_priority):
        prefix = item[2]['search_metrics']['contact_prefix']
        if any(
            abs(prefix['time_s'] - earlier['time_s']) < .4
            and np.linalg.norm(np.subtract(prefix['xy_m'], earlier['xy_m'])) < 1.
            and abs(prefix['speed_mps'] - earlier['speed_mps']) < .75
            and abs(prefix['acceleration_mps2'] - earlier['acceleration_mps2']) < .75
            and abs(prefix['yaw_rad'] - earlier['yaw_rad']) < .1
            and abs(prefix['steering_rad'] - earlier['steering_rad']) < .05
            for _, _, record in chosen
            for earlier in [record['search_metrics']['contact_prefix']]
        ):
            continue
        chosen.append(item)
        if len(chosen) == MAX_PREFIXES:
            break
    return chosen


def continuation_seeds(search, start, knots, first):
    """Keep the earlier controls fixed and explore several terminal speeds."""
    tail = knots >= first
    yield search.route_start(start, first)
    for acceleration, duration in [(0., 0.), (1.5, 1.5), (2.5, 1.5),
                                   (2.5, 3.), (2.5, np.inf), (-1., np.inf)]:
        values = start.copy()
        values[:search.n][tail] = np.where(
            knots[tail] < first + duration, acceleration, 0.)
        yield search.route_start(values, first)
        yield values


def continue_contact_prefixes(search, records, evaluate, limit, budget_reached):
    from vehicle_control_search import refinement_groups

    knots = search.control_times[search.editable]
    prefixes = select_prefixes(records, search.editor.constraints.get('official_margin_m', .1), knots[-1])
    if not prefixes or len(records) >= limit:
        return False

    def set_source(record, first):
        search.activate(record['initial_pose_adjustment'])
        search.continuation_source = {
            'attempt_id': record['attempt_id'],
            'first_edit_time_s': float(first),
        }

    def evaluate_seeds(start, first, maximum):
        seen = {tuple(start)}
        for values in continuation_seeds(search, start, knots, first):
            key = tuple(values)
            if key in seen:
                continue
            evaluate(values)
            seen.add(key)
            if len(records) >= maximum:
                break

    search.search_phase = 'continuation'
    # Give each prefix a chance before spending evaluations on fine adjustments.
    seed_budget = min(limit - len(records), (limit - len(records)) * 2 // 3 + 1)
    seed_end = len(records) + seed_budget
    branches = []
    for index, prefix_item in enumerate(prefixes):
        if len(records) >= seed_end:
            break
        _, start, record = prefix_item
        prefix_time = record['search_metrics']['contact_prefix']['time_s']
        first = knots[np.searchsorted(knots, prefix_time)]
        set_source(record, first)
        before = len(records)
        end = before + max(1, (seed_end - before) // (len(prefixes) - index))
        evaluate_seeds(start, first, end)
        best = min([prefix_item, *records[before:]], key=continuation_priority)
        branches.append((best, first))

    # If the continuation alone cannot clear the checks, vary one earlier knot.
    if not any(item[2]['search_metrics']['geometry_motion_guidance_feasible'] for item in records):
        search.search_phase = 'overlap'
        for item, first in sorted(branches, key=lambda branch: continuation_priority(branch[0]))[:2]:
            if len(records) >= limit:
                break
            earlier = knots[knots < first]
            if not len(earlier):
                continue
            _, start, record = item
            set_source(record, earlier[-1])
            before = len(records)
            evaluate_seeds(start, earlier[-1], min(limit, before + 6))
            if len(records) > before:
                branches.append((min(records[before:], key=continuation_priority), earlier[-1]))

    search.search_phase = 'continuation_refinement'
    finalists = sorted(branches, key=lambda branch: continuation_priority(branch[0]))[:2]
    for index, (item, first) in enumerate(finalists):
        _, start, record = item
        set_source(record, first)
        editable = np.flatnonzero(knots >= first)
        indices = np.r_[editable, editable + search.n]
        groups, bounds = refinement_groups(
            start[indices], search.lower[indices], search.upper[indices], len(editable))
        if not groups:
            continue
        local_end = len(records) + (limit - len(records)) // (len(finalists) - index)
        if local_end - len(records) <= len(groups):
            continue

        def residual(offsets):
            if len(records) >= local_end:
                raise budget_reached
            values = start.copy()
            for group, offset in zip(groups, offsets):
                values[indices[group]] += offset
            return evaluate(values.clip(search.lower, search.upper))

        try:
            least_squares(residual, np.zeros(len(groups)),
                          bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=12)
        except budget_reached:
            pass
    return True
