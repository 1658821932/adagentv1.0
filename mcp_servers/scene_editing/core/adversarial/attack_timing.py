"""Shared timing policy for sampled attack-scene checks."""

from acceptance_policy import OBSERVATION_END_S
MIN_ATTACK_TIME_S = OBSERVATION_END_S


def contact_after_observation(times, overlap_indices):
    """Reject any sampled ego overlap during or at the observation boundary."""
    return all(float(times[index]) > MIN_ATTACK_TIME_S for index in overlap_indices)
