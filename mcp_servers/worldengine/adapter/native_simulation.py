"""Run native closed loop with the coordinate adapter for exported edited scenes.

Keep WE's policy, controller and metric formulas; use the shared full scoring horizon.
The adapter shared with native replay fixes Gaussian/ego coordinate calibration
and keeps explicitly edited stationary actors represented in rendered states.
"""
from pathlib import Path
import json
import os
import pickle
import runpy
import sys
try:
    from .startup_timing import get_timing
except ImportError:
    from startup_timing import get_timing

_STARTUP = get_timing('simulator')
_STARTUP.mark('wrapper_entered')


def activate_exported_scene_adapter(scenes):
    exported = [str(key) for key, scene in scenes.items()
                if scene.get('metadata', {}).get('native_export')]
    if exported:
        for key in exported:
            reference=scenes[key]['metadata']['native_export']
            if reference.get('ego_calibration_frame') != 'reconstruction':
                raise ValueError('Legacy exported UTM ego calibration must be re-exported with MTGS reconstruction poses before simulation; reuse the saved trajectory, not a new Agent run')
        from replay_pose import install
        install()
    return exported


def main():
    # launch.py supplies the original WE entry followed by its exact overrides.
    entry = Path(sys.argv[1]).resolve()
    overrides = sys.argv[2:]
    values = dict(value.split('=', 1) for value in overrides if '=' in value)
    def path_value(key):
        value=values[key]
        return Path(json.loads(value) if value.startswith('"') else value)
    source = path_value('data_file_path')
    output = path_value('output_dir')
    with _STARTUP.span('scenario_load'):
        with source.open('rb') as stream:
            scenes = pickle.load(stream)
    with _STARTUP.span('scene_adapter_and_framework_imports'):
        exported = activate_exported_scene_adapter(scenes)
    audit = dict(adapter='edited_scene_coordinates.v1', exported_scene_ids=exported,
                 pose_adapter_active=bool(exported), scenario_file=str(source),
                 policy_and_metrics_unchanged=True, status='running')
    output.mkdir(parents=True, exist_ok=True)
    if values.get('with_metric_manager')=='true':
        from evaluation_horizon import full_horizon,install_metric_audit
        history=int(values['num_history']);future=int(values['num_future'])
        horizon=full_horizon(history,history+future,int(values['sampling_poses']))
        if int(values['buffer_size'])!=horizon['buffer_size']:
            raise ValueError('PDM buffer_size must match the derived full horizon')
        install_metric_audit(output,horizon)
    collision_target=os.getenv('ADAGENT_COLLISION_TARGET_TRACK')
    if collision_target:
        from collision_report import install_recorder
        install_recorder(output/'native_collision_states.jsonl')
    report = output/'native_scene_adapter.json'
    report.write_text(json.dumps(audit, indent=2))
    sys.argv = [str(entry), *overrides]
    try:
        with _STARTUP.span('native_entry_total'):
            runpy.run_path(str(entry), run_name='__main__')
        if exported:
            from worldengine.engine.engine_utils import get_engine
            engine = get_engine()
            manager = None if engine is None else engine.managers.get('data_manager')
            if manager is None or not manager.episode_data:
                # launch.py 在多进程模式下跑仿真：帧数据在子进程里，
                # 父进程本来就没有，这不是失败。碰撞状态已写进 jsonl。
                audit['collected_frames'] = 'not_in_parent_process'
            else:
                # Closed loop may legitimately terminate early. Save actual frames;
                # unlike trajectory replay, do not require a full-length episode.
                manager.save_data()
                audit['last_episode_frame_count'] = len(manager.episode_data)
        audit['status'] = 'returned'
    except BaseException as error:
        audit.update(status='error', error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        # 碰撞报告只依赖落盘的 jsonl，放在 finally 里，仿真成功与否都尝试生成
        if collision_target:
            try:
                from collision_report import write_report
                collision=write_report(output/'native_collision_states.jsonl',output/'native_collision_report.json',
                    collision_target,int(values['num_history'])-1)
                audit['collision_report_path']=collision['report_path']
            except BaseException as collision_error:
                audit['collision_report_error']=type(collision_error).__name__+': '+str(collision_error)[:300]
        report.write_text(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
