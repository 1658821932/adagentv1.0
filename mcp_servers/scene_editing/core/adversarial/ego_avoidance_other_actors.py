"""Full-window screening of an ego candidate against recorded other vehicles."""
import json
from pathlib import Path

import numpy as np

from ego_avoidance_checks import evaluate_reference
from other_actor_checks import screen_other_actors


def crop_context(context, times_s):
    """Crop a recorded context without filling or renumbering observations."""
    times_s = np.asarray(times_s, dtype=float)
    if context.get('version') != 'recorded_other_vehicles.v1':
        raise ValueError('Unsupported other-vehicle context')
    source_times = np.asarray(context.get('times_s'), dtype=float)
    if not len(times_s) or len(source_times) < len(times_s):
        raise ValueError('Recorded other-vehicle context does not cover the ego window')
    np.testing.assert_allclose(source_times[:len(times_s)], times_s, rtol=0, atol=1e-9)
    actors = []
    dropped = []
    for actor in context.get('actors', []):
        samples = [sample for sample in actor.get('samples', [])
                   if int(sample['sample_index']) < len(times_s)]
        if samples:
            actors.append(dict(track_id=int(actor['track_id']), samples=samples))
        else:
            dropped.append(int(actor['track_id']))
    cropped = dict(context, times_s=times_s.tolist(), actors=actors,
        source_window_samples=len(source_times), cropped_window_samples=len(times_s),
        actors_without_observations_in_window=dropped,
        crop_policy='Keep original sample indices and observed samples only; no interpolation or extrapolation')
    # Validate identities, timestamps and geometry through the common screener.
    if actors:
        screen_other_actors(times_s, np.full((len(times_s), 2), 1e6), np.zeros(len(times_s)),
            4., 2., cropped, first_ego_contact=None)
    return cropped


def planning_view(scene):
    dt = float(np.median(np.diff(scene.times_s)))
    if np.isclose(dt, .05, rtol=0, atol=1e-9):
        return scene.planning_view()
    if np.isclose(dt, .1, rtol=0, atol=1e-9):
        return scene
    raise ValueError('Expected a canonical 20 Hz or planning 10 Hz frozen scene')


def load_context(path, scene):
    context = json.loads(Path(path).read_text())
    planning = planning_view(scene)
    if int(context.get('target_track_id', -1)) != int(planning.metadata['target_track_id']):
        raise ValueError('Other-vehicle context target differs from the frozen attack')
    return crop_context(context, planning.times_s)


def evaluate_candidate(scene, trajectory, limits, other_context=None):
    """Run the common ego-attack gates, then all observed other-vehicle gates."""
    planning = planning_view(scene)
    geometry = planning.metadata['geometry']
    result = evaluate_reference(trajectory.times_s, trajectory.xy, trajectory.heading_rad,
        planning.attack_xy, planning.attack_heading_rad,
        geometry['ego']['length_m'], geometry['ego']['width_m'],
        geometry['target']['length_m'], geometry['target']['width_m'], limits,
        planning.ego_xy, planning.ego_heading_rad, task_policy=scene.metadata.get('acceptance_policy'))
    if other_context is None:
        result['other_actor_check'] = dict(status='not_checked')
        return result
    other = screen_other_actors(trajectory.times_s, trajectory.xy, trajectory.heading_rad,
        geometry['ego']['length_m'], geometry['ego']['width_m'], other_context,
        first_ego_contact=None)
    result['other_actor_check'] = other
    result['violations'] = list(result['violations']) + list(other['violations'])
    result['reference_passed'] = bool(result['reference_passed'] and other['status'] == 'passed')
    result['status'] = 'reference_passed' if result['reference_passed'] else 'reference_failed'
    return result
