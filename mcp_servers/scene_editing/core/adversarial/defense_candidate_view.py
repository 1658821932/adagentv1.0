"""Unranked decision evidence; no changes to trajectory checks or numeric archives."""
from math import hypot

VERSION = 'unranked_motion.v1'
RANK_FIELDS = frozenset((
    'ranking_scope', 'best_labels_note', 'best', 'best_candidate_id',
    'constructor_preferred_candidate_id', 'constructor_preferred_clearance_m',
    'constructor_best_candidate_id', 'constructor_best_clearance_m',
    'constructor_best_by_progress_candidate_id', 'constructor_best_by_progress_ratio',
    'pool_preferred_candidate_id', 'pool_preferred_clearance_m', 'pool_best_candidate_id',
    'pool_best_clearance_m', 'pool_best_by_progress_candidate_id', 'pool_best_by_progress_ratio',
    'foci_best_candidate_id', 'best_min_target_clearance_m', 'best_change_reason', 'is_new_best',
    'recommended_next_parent_id', 'recommended_parent_preview', 'delta_vs_previous_best_m',
    'profile_vs_previous_best', 'best_profile_vs_constructor', 'best_profile_vs_pool',
    'agent_improvement_over_constructor_m', 'agent_improvement_over_pool_m',
    'foci_improvement_over_constructor_m', 'agent_pareto_improvement_over_constructor',
    'agent_pareto_improvement_over_pool', 'productive_edit', 'consecutive_non_improving_edits_on_branch',
    'dominates_parent', 'delta_vs_parent_m'))

GUIDANCE = (
    'Candidates are listed in generation order, with no quality ranking, preferred ID or overall score. '
    'Choose explicitly using the task, passed/failed checks and motion evidence. A clearance-curve dominance label concerns separation only, not an overall improvement in progress or driving quality. '
    'Compare clearance during the editable period and the recorded-ego contact samples; a minimum in '
    'fixed observation history does not measure the benefit of a defense. '
    'Keep an evidenced safety reserve for approximation/model uncertainty: do not seek the smallest gap '
    'or delay braking until contact. Also assess whether early stopping or strong braking is needed '
    'when a milder checked alternative has adequate reserve. There is no universal added gap threshold, '
    'minimum progress target or required number of edits. '
    'Read the clearance and speed timelines, actual braking start/stop states, other-actor and road checks, '
    'terminal speed and closing rate. A positive terminal speed with a shrinking gap deserves explicit '
    'attention; checks cover the observed horizon only. Small acceleration differences alone do not '
    'establish a meaningful comfort difference. Native sampled acceleration and analytic curve bounds '
    'have different scopes. '
    'An unchanged, constructor or FOCI candidate may be selected; FOCI output is not automatically better '
    'than its parent. If the chosen plan has a concrete weakness and a supported improvement remains, '
    'use bounded construction, local FOCI or editing to compare it, preserving a passing fallback. '
    'Explain the chosen safety/behavior tradeoff with actual candidate evidence, without inventing '
    'that continued travel is unsafe when checked moving alternatives have not been considered. '
    'The frozen attack scene fixes the attack trajectory identity; its vehicle can still move over time.')


def unranked_view(value):
    """Remove recommendation signals recursively, retaining order and check evidence."""
    if isinstance(value, dict):
        return {key: unranked_view(item) for key, item in value.items() if key not in RANK_FIELDS}
    if isinstance(value, list):
        return [unranked_view(item) for item in value]
    return value


def temporal_evidence(times, xy, clearance, boundary, contact_indices=(), stop_speed=.1):
    """Native-frame diagnostics; source timestamps and hard checks remain unchanged."""
    times=[float(t) for t in times];gaps=[float(g) for g in clearance]
    if len(times)!=len(xy) or len(times)!=len(gaps) or len(times)<2:
        raise ValueError('Motion evidence requires aligned native samples')
    speed=[hypot(float(b[0]-a[0]),float(b[1]-a[1]))/(t1-t0)
           for a,b,t0,t1 in zip(xy,xy[1:],times,times[1:])]
    def period(indices):
        if not indices:
            return None
        i=min(indices,key=lambda index:gaps[index])
        return dict(sample_count=len(indices),sample_time_range_s=[times[indices[0]],times[indices[-1]]],
                    min_clearance_m=round(gaps[i],3),minimum_time_s=times[i],
                    minimum_frame_index=i)
    editable=[i for i,t in enumerate(times) if t>boundary]
    observation=[i for i,t in enumerate(times) if t<=boundary]
    contacts=[i for i in contact_indices if times[i]>boundary]
    last_moving=max((i for i,v in enumerate(speed) if v>stop_speed),default=-1)
    stopped=speed[-1]<=stop_speed
    stop_index=last_moving+1 if stopped else None
    editable_travel=sum(v*(t1-max(t0,boundary)) for v,t0,t1 in zip(speed,times,times[1:]) if t1>boundary)
    return dict(
        clearance_periods=dict(observation=period(observation),editable=period(editable),
            recorded_ego_contact_samples=period(contacts)),
        motion_timeline=dict(time_basis='frame_times_s; speed[i] covers frame i to i+1',
            target_clearance_m=[round(g,3) for g in gaps],
            speed_to_next_mps=[round(v,3) for v in speed]),
        motion_outcome=dict(editable_travel_m=round(editable_travel,3),terminal_stopped=stopped,
            terminal_stop_first_sample_s=None if stop_index is None else times[stop_index],
            terminal_xy_m=[round(float(v),3) for v in xy[-1]],
            scope='Sampled motion and body clearance; contact-period samples come from recorded ego against the same fixed attack. Not an acceptance window or an added safety threshold.'))


BEHAVIOR_GUIDANCE = (
    'Driving objective: respect the specific user target_effect/maneuver and preserve the fixed observation prefix. Subject to all contact, other-actor, road '
    'and motion checks and an evidenced safety reserve, retain effective forward progress along the '
    'original route and reasonably normal driving. Avoid unnecessarily early braking, strong braking '
    'or stopping far behind the attack; do not trade reserve for progress or seek a minimum gap. '
    'Extra clearance has no automatic value once the needed reserve is provided. '
    'behavior_metrics projects NET advancement onto the original route after the history boundary; '
    'reverse and lateral detours cannot earn progress. progress_ratio_of_recorded is the legacy total '
    'path-length diagnostic and is NOT the new objective. '
    'Reserve parameters are explicit research optimization preferences, NOT calibrated prediction or '
    'execution error bounds, legal following distances or new acceptance gates. A shortfall is a '
    'reason for closer assessment, not permission to submit an unsafe path or a new automatic rejection. '
    'Use candidate_differences rather than mental subtraction. When choosing lower progress over a '
    'passed alternative with comparable reserve, identify the concrete evidenced risk avoided and '
    'whether its benefit warrants the loss. A generic preference for larger clearance or unknown '
    'simulation error alone does not establish that the other candidate is unsafe. '
    'If a safer useful refinement is supported, use the remaining generation tools; otherwise explain '
    'the actual tradeoff. No fixed progress percentage, edit count or preferred candidate is imposed.')
