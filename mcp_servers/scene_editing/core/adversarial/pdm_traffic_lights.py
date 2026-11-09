"""Explicit traffic-light simplification for PDM experiments; annotations stay intact."""


def traffic_light_policy(frames_by_scene, *, ignore=False):
    counts = {
        name: dict(frames_with_signals=sum(bool(frame.get('traffic_lights')) for frame in frames),
                   signal_entries=sum(len(frame.get('traffic_lights') or []) for frame in frames))
        for name, frames in frames_by_scene.items()
    }
    if not ignore and any(value['frames_with_signals'] for value in counts.values()):
        raise ValueError('traffic light conversion required; explicit ignore mode is disabled')
    return dict(mode='ignored' if ignore else 'require_empty',
                signals_passed_to_planner=False, source_annotation_counts=counts,
                traffic_light_compliance='not_evaluated', source_annotations_modified=False)
