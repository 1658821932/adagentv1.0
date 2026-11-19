"""Shared timing policy for sampled attack-scene checks."""

from acceptance_policy import OBSERVATION_END_S
MIN_ATTACK_TIME_S = OBSERVATION_END_S


def contact_after_observation(times, overlap_indices, earliest_contact_s=MIN_ATTACK_TIME_S, time_tolerance_s=0.):
    """Use the current history default; archived policies pass their own boundary."""
    if time_tolerance_s>0:
        return all(float(times[index]) >= earliest_contact_s-time_tolerance_s for index in overlap_indices)
    return all(float(times[index]) > earliest_contact_s for index in overlap_indices)
