"""Native WE state replay with MTGS rendering in the existing render environment.

The simulator continues to own every agent pose. Only GPU rendering crosses the
process boundary, avoiding an incompatible simulator/gsplat binary stack.
"""
import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import numpy as np


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported renderer message value '+type(value).__name__)


class NativeRenderBridge:
    def __init__(self,**options):
        self.options=options;self.process=None;self.sequence=0
        self.request=json.loads(Path(os.environ['WE_NATIVE_RENDER_REQUEST']).read_text())
        self.folder=Path(self.request['output_dir'])/'gpu_renderer';self.folder.mkdir(exist_ok=False)
        self.log=(self.folder/'worker.log').open('x')
        atexit.register(self.close)

    def reset(self,scene_id,**kwargs):
        if self.process is not None:raise ValueError('Native GPU worker is bound to exactly one scene')
        config=dict(self.request,renderer_options=self.options,scene_id=scene_id,renderer_output_dir=str(self.folder))
        path=self.folder/'request.json';path.write_text(json.dumps(config))
        self.process=subprocess.Popen([self.request['render_python'],'-B',str(Path(__file__).with_name('replay_gpu_worker.py')),'--request',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env=dict(os.environ))
        ready=self._response()
        if ready.get('status')!='ready':raise RuntimeError('GPU renderer did not initialize; inspect '+str(self.folder/'worker.log'))

    def _response(self):
        selector=selectors.DefaultSelector();selector.register(self.process.stdout,selectors.EVENT_READ)
        try:
            if not selector.select(180):raise TimeoutError('Native GPU rendering timed out; inspect '+str(self.folder/'worker.log'))
            line=self.process.stdout.readline()
        finally:selector.close()
        if not line:raise RuntimeError('Native GPU renderer exited; inspect '+str(self.folder/'worker.log'))
        result=json.loads(line)
        if result.get('status')=='error':raise RuntimeError('Native GPU renderer failed: '+result['error']+'; inspect '+str(self.folder/'worker.log'))
        return result

    def render(self,render_state):
        import cv2
        self.process.stdin.write(json.dumps(dict(sequence=self.sequence,render_state=render_state),default=json_value)+'\n');self.process.stdin.flush()
        response=self._response()
        if response.get('sequence')!=self.sequence:raise RuntimeError('GPU frame response identity differs')
        cameras={}
        for name,path in response['camera_images'].items():
            frame=Path(path).resolve()
            if not frame.is_relative_to(self.folder.resolve()):raise ValueError('GPU response image is outside this replay')
            image=cv2.imread(str(frame))
            if image is None:raise RuntimeError('GPU renderer image missing or unreadable')
            cameras[name]=dict(image=image)
        self.sequence+=1
        return dict(cameras=cameras,lidars={},ego2global=np.array(response['ego2global']),render_state=render_state)

    def close(self):
        process=self.process
        if process is not None and process.poll() is None:
            try:
                process.stdin.write('{"stop":true}\n');process.stdin.flush();process.wait(timeout=10)
            except (OSError,subprocess.TimeoutExpired):process.terminate();process.wait(timeout=10)
        if not self.log.closed:self.log.close()

    def destroy(self):self.close()
