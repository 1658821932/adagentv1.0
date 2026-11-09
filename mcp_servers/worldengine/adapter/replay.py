"""Native replay entry separate from policy inference and legacy editing renders."""
import json
import os
from pathlib import Path
import subprocess
from mcp_servers.runtime import runtime_task, current_run, write_json
from mcp_servers.process import run_logged
from .config import WorldEngineConfig, SIMENGINE_PYTHON, PROJECT_ROOT, SHARED_LIBRARY_DIR, FFMPEG, REPLAY_RENDER_PYTHON, NUPLAN_SOURCE, GSPLAT_EXTENSION


def decode_video(path):
    executable=str(FFMPEG)
    decoded=subprocess.run([executable,'-v','error','-xerror','-i',str(path),'-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True,timeout=120)
    if decoded.returncode:raise RuntimeError('Replay video decoding failed: '+str(path))
    counts=[int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=') and line.split('=',1)[1].strip().isdigit()]
    if not counts or max(counts)<1:raise RuntimeError('Replay video contains no readable frames: '+str(path))
    return max(counts)


@runtime_task('worldengine','replay')
def run_replay(arguments):
    runtime=WorldEngineConfig()
    package=arguments['simulation_input']
    folder=Path(package['scenario_data_folder_path']).expanduser()
    assets=Path(package['asset_folder_path']).expanduser()
    scenario=folder/package.get('scenario_pkl_file_name','all_scenarios.pkl')
    if not scenario.is_file():raise FileNotFoundError('Prepared native replay scenario missing: '+str(scenario))
    if not assets.is_dir():raise FileNotFoundError('Native replay assets missing: '+str(assets))
    python=Path(SIMENGINE_PYTHON)
    if not python.is_file():raise FileNotFoundError('SimEngine interpreter missing: '+str(python))
    task=current_run('worldengine')
    output=task.output/'replay'
    request=dict(arguments,ffmpeg=str(FFMPEG),render_python=str(REPLAY_RENDER_PYTHON),nuplan_source=str(NUPLAN_SOURCE),gsplat_extension=str(GSPLAT_EXTENSION),worldengine_root=str(runtime.project_root),scenario_pkl=str(scenario.resolve()),
        asset_folder_path=str(assets.resolve()),staged_scenario=str(task.input/'replay_scenario.pkl'),
        output_dir=str(output),report_path=str(task.output/'replay-report.json'))
    request_path=task.input/'replay-request.json';write_json(request_path,request)
    env=task.child_env()
    env['WORLDENGINE_ROOT']=str(runtime.project_root)
    if SHARED_LIBRARY_DIR.is_dir():
        env['LD_LIBRARY_PATH']=str(SHARED_LIBRARY_DIR)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    env['PYTHONPATH']=str(runtime.project_root/'projects/SimEngine')+os.pathsep+env.get('PYTHONPATH','')
    completed=run_logged([str(python),'-B',str(Path(__file__).with_name('replay_native.py')),
        '--request',str(request_path)],cwd=runtime.project_root,env=env,
        log_path=task.logs/'replay.log',timeout=arguments.get('timeout',1800))
    if completed.returncode:raise RuntimeError('Native WE replay failed; inspect '+str(task.logs/'replay.log'))
    report_path=task.output/'replay-report.json'
    if not report_path.is_file():raise RuntimeError('Native replay produced no completion report')
    report=json.loads(report_path.read_text())
    if report.get('status')!='completed' or report.get('scene_id')!=arguments['scene_id']:
        raise RuntimeError('Native replay did not complete the requested scene')
    videos=sorted(output.rglob('*.mp4'))
    if not videos:raise RuntimeError('Native WE replay produced no video; inspect '+str(task.logs/'replay.log'))
    frames={str(p):decode_video(p) for p in videos}
    report['video_frames']=frames
    write_json(report_path,report)
    return dict(mode='real',scene_id=arguments['scene_id'],result_dir=str(output),videos=list(frames),
                replay_report=str(report_path),checks={'native_replay':'passed','video_decode':'passed'})
