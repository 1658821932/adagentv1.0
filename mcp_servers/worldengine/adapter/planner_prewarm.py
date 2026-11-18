"""Local, opt-out framework preloading for editor-owned VAD probes.

Only imports survive between jobs. Each job builds a fresh policy from the
original native entry, keeping temporal state, config and weights isolated.
The private Unix socket never receives API credentials or model inputs.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time


def _write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=str(path.parent),prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as handle:json.dump(value,handle,allow_nan=False)
        os.replace(name,str(path))
    finally:
        if os.path.exists(name):os.unlink(name)


def _read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def _alive(state):
    try:
        fields=Path('/proc/{}/stat'.format(int(state['pid']))).read_text().rsplit(')',1)[1].split()
        return fields[0]!='Z' and fields[19]==state['process_start']
    except (KeyError,OSError,ValueError):return False


def _environment(source):
    # A resident native worker must not inherit host Agent API credentials.
    names={'PATH','HOME','USER','LOGNAME','LANG','TMPDIR','LD_LIBRARY_PATH','PYTHONPATH',
           'CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONUNBUFFERED','PYTHONHASHSEED'}
    prefixes=('LC_','OMP_','MKL_','OPENBLAS_','WORLDENGINE_','ADAGENT_')
    return {key:value for key,value in source.items() if key in names or key.startswith(prefixes)}


def _layout(policy):
    from .config import ALGENGINE_PYTHON
    root=Path(policy['worldengine_root']).resolve()
    entry=root/'projects/AlgEngine/closed_loop/sim_test.py'
    interpreter=Path(os.environ.get('ALGENGINE_PYTHON') or ALGENGINE_PYTHON).absolute()
    stat=entry.stat()
    worker_stat=Path(__file__).stat()
    artifacts=[]
    for name in ('config_path','checkpoint_path'):
        if policy.get(name):
            path=Path(policy[name]).resolve();metadata=path.stat()
            artifacts.append((str(path),metadata.st_size,metadata.st_mtime_ns))
    identity=json.dumps([str(root),str(interpreter),str(entry),stat.st_size,stat.st_mtime_ns,
                         worker_stat.st_size,worker_stat.st_mtime_ns,
                         artifacts,os.environ.get('CUDA_VISIBLE_DEVICES')],sort_keys=True)
    key=hashlib.sha256(identity.encode()).hexdigest()[:20]
    parent=Path(tempfile.gettempdir())/('adagent-vad-prewarm-'+str(os.getuid()))
    folder=parent/key
    return root,entry,interpreter,folder


def locate(policy):
    """Return an existing compatible worker; never start work during a lookup."""
    if os.environ.get('ADAGENT_VAD_PREWARM_ENABLED','true').lower() in ('false','0','off'):
        return None
    try:_,_,_,folder=_layout(policy)
    except (OSError,KeyError):return None
    state=_read(folder/'state.json')
    if state.get('status') in ('warming','ready','busy') and _alive(state):
        return dict(socket_path=str(folder/'worker.sock'),state_path=str(folder/'state.json'),**state)
    return None


def start(policy, report_path):
    """Start asynchronously before selection. Failure preserves native cold launch."""
    if os.environ.get('ADAGENT_VAD_PREWARM_ENABLED','true').lower() in ('false','0','off'):
        result=dict(status='disabled');_write(report_path,result);return result
    try:
        from .config import runtime_env,PROJECT_ROOT
        root,entry,interpreter,folder=_layout(policy)
        folder.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        folder.mkdir(exist_ok=True,mode=0o700)
        os.chmod(str(folder.parent),0o700);os.chmod(str(folder),0o700)
        with (folder/'start.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            existing=locate(policy)
            if existing:
                result=dict(existing,reused=True)
            else:
                env=_environment(runtime_env(os.environ))
                env.update(WORLDENGINE_ROOT=str(root),PYTHONUNBUFFERED='1',
                    PYTHONPATH=os.pathsep.join([str(root/'projects/AlgEngine'),str(root/'projects/SimEngine'),str(PROJECT_ROOT)]),
                    ADAGENT_STARTUP_TIMING_DIR=str(folder/'startup_timing'))
                with (folder/'worker.log').open('ab') as log:
                    process=subprocess.Popen([str(interpreter),str(Path(__file__).resolve()),'serve',
                        '--entry',str(entry),'--folder',str(folder)],cwd=str(root/'projects/AlgEngine'),
                        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                # The initial record closes the small launch/import race for reuse.
                fields=Path('/proc/{}/stat'.format(process.pid)).read_text().rsplit(')',1)[1].split()
                state=dict(status='warming',pid=process.pid,process_start=fields[19],started_at=time.time(),
                    entry=str(entry),scope='framework_imports_only_fresh_model_per_candidate')
                _write(folder/'state.json',state)
                result=dict(state,socket_path=str(folder/'worker.sock'),state_path=str(folder/'state.json'),reused=False)
        _write(report_path,result)
        return result
    except Exception as error:
        result=dict(status='unavailable',error_type=type(error).__name__,fallback='native_cold_launch')
        _write(report_path,result)
        return result


def _send(connection, value):
    connection.sendall(json.dumps(value,allow_nan=False).encode()+b'\n')


def _receive(connection):
    data=bytearray()
    while not data.endswith(b'\n'):
        chunk=connection.recv(1)
        if not chunk:raise EOFError('Prewarm connection closed')
        data.extend(chunk)
        if len(data)>1024*1024:raise ValueError('Prewarm command exceeds limit')
    return json.loads(data)


def _cancel_on_disconnect(connection, finished):
    # Cancellation of launch/client must also terminate an active resident job.
    while not finished.wait(.25):
        try:
            readable,_,_=select.select([connection],[],[],0)
            if readable and not connection.recv(1,socket.MSG_PEEK):
                os.kill(os.getpid(),signal.SIGTERM)
                return
        except OSError:return


def _execute(module, base_monitor, entry, command, environment):
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-frames',type=int,required=True)
    args,forwarded=parser.parse_known_args(command)
    if len(forwarded)<3 or Path(forwarded[0]).resolve()!=entry or args.scene_frames<5:
        raise ValueError('Prewarm job must use the configured native planner entry')
    class ConfiguredMonitor(base_monitor):
        def __init__(self,folder_path,check_interval=1.0,max_history=4,max_frame_per_scene=12,**options):
            super().__init__(folder_path=folder_path,check_interval=check_interval,max_history=max_history,
                max_frame_per_scene=args.scene_frames,**options)
    module.AsyncFileMonitor=ConfiguredMonitor
    previous_env=dict(os.environ);previous_argv=sys.argv[:]
    try:
        os.environ.update(_environment(environment))
        from startup_timing import get_timing
        module._STARTUP=get_timing('planner')
        module._STARTUP.mark('framework_cache_reused')
        sys.argv=[str(entry),*forwarded[1:]]
        # Native main loads the real per-candidate config/weights and a fresh model.
        with module._STARTUP.span('native_entry_total'):module.main()
    finally:
        sys.argv=previous_argv;os.environ.clear();os.environ.update(previous_env)
        # Model/cfg locals have left main; free GPU allocations between scenes.
        gc.collect()
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()
        import logging
        logger=logging.getLogger('mmdet')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler);handler.close()
        try:
            from mmcv.utils.logging import logger_initialized
            logger_initialized.pop('mmdet',None)
        except ImportError:pass


def serve(entry, folder, idle_seconds=900):
    folder=Path(folder);entry=Path(entry).resolve();socket_path=folder/'worker.sock'
    # start() publishes ownership first so a concurrent caller can reuse warming.
    for _ in range(100):
        state=_read(folder/'state.json')
        if state.get('pid')==os.getpid():break
        time.sleep(.01)
    started=time.perf_counter()
    try:
        spec=importlib.util.spec_from_file_location('_adagent_prewarm_native_planner',str(entry))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        base_monitor=module.AsyncFileMonitor
        state.update(status='ready',framework_import_seconds=round(time.perf_counter()-started,3),
                     ready_at=time.time(),completed_jobs=0)
        socket_path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(socket_path));os.chmod(str(socket_path),0o600);server.listen(4)
            server.settimeout(idle_seconds);_write(folder/'state.json',state)
            while True:
                try:connection,_=server.accept()
                except socket.timeout:break
                with connection:
                    command=_receive(connection)
                    if command.get('op')=='status':_send(connection,state);continue
                    if command.get('op')=='shutdown':_send(connection,dict(status='stopped'));break
                    if command.get('op')!='run':raise ValueError('Unknown prewarm operation')
                    state.update(status='busy',job_started_at=time.time());_write(folder/'state.json',state)
                    _send(connection,dict(status='accepted',pid=os.getpid()))
                    finished=threading.Event()
                    threading.Thread(target=_cancel_on_disconnect,args=(connection,finished),daemon=True).start()
                    try:
                        _execute(module,base_monitor,entry,command['argv'],command.get('environment',{}))
                    except BaseException as error:
                        finished.set()
                        _send(connection,dict(status='failed',error_type=type(error).__name__))
                        raise
                    else:
                        finished.set();state.update(status='ready',completed_jobs=state['completed_jobs']+1)
                        _write(folder/'state.json',state)
                        _send(connection,dict(status='completed',pid=os.getpid(),completed_jobs=state['completed_jobs']))
        state['status']='stopped'
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__)
        raise
    finally:
        _write(folder/'state.json',state);socket_path.unlink(missing_ok=True)


def run_client(socket_path, original, ready_timeout=600):
    socket_path=Path(socket_path);state_path=socket_path.parent/'state.json'
    deadline=time.monotonic()+ready_timeout
    while True:
        state=_read(state_path)
        if not _alive(state) or state.get('status') in ('failed','stopped'):
            # No job was accepted: the original cold path is still authoritative.
            os.execv(original[0],original)
        if socket_path.exists():break
        if time.monotonic()>=deadline:raise TimeoutError('Framework prewarm did not become ready')
        time.sleep(.1)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.connect(str(socket_path))
        _send(connection,dict(op='run',argv=original[2:],environment=_environment(os.environ)))
        accepted=_receive(connection)
        if accepted.get('status')!='accepted':raise RuntimeError('Prewarm job was not accepted')
        # Once accepted, never replay a failed/partial native evaluation cold.
        result=_receive(connection)
        if result.get('status')!='completed':raise RuntimeError('Prewarmed native planner failed: '+str(result.get('error_type')))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='operation',required=True)
    worker=sub.add_parser('serve');worker.add_argument('--entry',required=True);worker.add_argument('--folder',required=True)
    client=sub.add_parser('client');client.add_argument('--socket',required=True)
    client.add_argument('--ready-timeout',type=float,default=600);client.add_argument('original',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.operation=='serve':serve(args.entry,args.folder)
    else:run_client(args.socket,args.original[1:] if args.original[:1]==['--'] else args.original,args.ready_timeout)


if __name__=='__main__':main()
