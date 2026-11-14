"""Shared timing policy for sampled attack-scene checks."""

from acceptance_policy import OBSERVATION_END_S
MIN_ATTACK_TIME_S = OBSERVATION_END_S


def contact_after_observation(times, overlap_indices, earliest_contact_s=MIN_ATTACK_TIME_S):
    """Use the current history default; archived policies pass their own boundary."""
    return all(float(times[index]) > earliest_contact_s for index in overlap_indices)
