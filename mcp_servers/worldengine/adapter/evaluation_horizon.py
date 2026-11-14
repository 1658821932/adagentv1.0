"""Score the configured rollout: standard native window is 1.5-7.5s at 2 Hz.

Explicit diagnostic budgets and editor probes still derive their full configured
rollout horizon; 'full_rollout_pdm' does not imply the whole source scene.
"""
import json
from pathlib import Path

DEFAULT_NATIVE_HISTORY = 4
DEFAULT_NATIVE_FRAMES = 16


def full_horizon(history,frames,sampling_poses=None):
    if type(history) is not int or type(frames) is not int or not 1<=history<frames:
        raise ValueError('Full-rollout PDM requires positive history and a later scene end')
    future=frames-history
    if sampling_poses is not None and sampling_poses!=future:
        raise ValueError('Full-rollout PDM sampling_poses is derived from scene length; omit the stale override')
    return dict(scope='full_rollout_pdm',num_history=history,max_frame_per_scene=frames,
        sampling_poses=future,buffer_size=future+1,start_step=history-1,end_step=frames-1,
        standard_4_second_benchmark=False)


def validate_metric_inputs(manager,horizon):
    count=horizon['buffer_size']
    actual=dict(ego_state_count=len(manager.get_pred_states()),
        reference_state_count=len(manager._cached_expert_traj),
        velocity_count=len(manager._cached_ego_velocities),
        obstacle_frame_count=len(manager.detection_tracks_list))
    if any(value!=count for value in actual.values()):
        raise RuntimeError('Full-rollout PDM sequence lengths differ: '+str(actual)+'; expected '+str(count))
    if (manager._proposal_sampling.num_poses!=horizon['sampling_poses'] or
            manager.buffer_size!=count or manager.current_step!=horizon['end_step']):
        raise RuntimeError('PDM did not evaluate the configured full rollout')
    return actual


def install_metric_audit(output,horizon):
    from worldengine.manager.metric_manager import MetricManager
    original=MetricManager.compute_pdm_scores
    rows=[]
    path=Path(output)/'pdm_evaluation_scope.json'
    def compute(manager,expert_trajectory,input_trajectory_velocities):
        actual=validate_metric_inputs(manager,horizon)
        interval=float(manager.current_scene['sample_rate'])*.05
        if abs(interval-.5)>1e-8:
            raise RuntimeError('This native PDM implementation requires 0.5s samples; explicitly resample before evaluation')
        result=original(manager,expert_trajectory,input_trajectory_velocities)
        rows.append(dict(horizon,scene_id=str(manager.current_scene['id']),status='completed',
            start_time_s=horizon['start_step']*interval,end_time_s=horizon['end_step']*interval,
            interval_s=interval,sequence_counts=actual))
        path.write_text(json.dumps(dict(status='completed',scope='full_rollout_pdm',
            standard_4_second_benchmark=False,scenes=rows),indent=2)+'\n')
        return result
    MetricManager.compute_pdm_scores=compute
