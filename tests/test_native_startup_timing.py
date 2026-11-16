"""Cold-start profiling preserves call order, errors and normal execution."""
import ast
import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from mcp_servers.worldengine.adapter.startup_timing import StartupTiming, get_timing, summarize

ROOT=Path(__file__).resolve().parents[1]


def test_nested_timing_preserves_return_and_error_without_recording_payload(tmp_path):
    timer=StartupTiming('planner',tmp_path)
    with timer.span('outer'):
        with timer.span('checkpoint_load'):
            value={'private_payload':'not passed into profiler'}
    with pytest.raises(ValueError,match='original error'):
        with timer.span('failure'):
            raise ValueError('original error')
    report=json.loads(summarize(tmp_path).read_text())
    phases=report['processes'][0]['phases']
    assert [p['status'] for p in phases]==['completed','completed','failed']
    assert phases[-1]['error_type']=='ValueError'
    assert all(p['elapsed_seconds']>=0 for p in phases)
    assert 'private_payload' not in json.dumps(report) and value['private_payload']


def test_disabled_or_unwritable_profiling_does_not_break_work(tmp_path,monkeypatch):
    monkeypatch.delenv('ADAGENT_STARTUP_TIMING_DIR',raising=False)
    with get_timing('offline').span('work'):pass
    assert not list(tmp_path.iterdir())
    blocked=tmp_path/'file';blocked.write_text('occupied')
    with pytest.warns(RuntimeWarning,match='simulation continues'):
        with StartupTiming('planner',blocked).span('work'):pass


def test_separate_child_processes_have_distinct_traces(tmp_path):
    folder=tmp_path/'timing'
    script="from mcp_servers.worldengine.adapter.startup_timing import get_timing; t=get_timing('simulator'); t.mark('wrapper_entered')"
    env=dict(os.environ,ADAGENT_STARTUP_TIMING_DIR=str(folder),PYTHONPATH=str(ROOT))
    for _ in range(2):
        result=subprocess.run([sys.executable,'-c',script],env=env,cwd=ROOT,capture_output=True,text=True,timeout=20)
        assert result.returncode==0,result.stderr
    report=json.loads(summarize(folder).read_text())
    assert len(report['processes'])==2
    assert len({p['pid'] for p in report['processes']})==2


def test_inherited_recorder_restarts_after_fork(tmp_path,monkeypatch):
    timer=StartupTiming('simulator',tmp_path);timer.mark('parent')
    parent=timer.pid;monkeypatch.setattr(os,'getpid',lambda:parent+1000000)
    timer.mark('child')
    reports=[json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert {tuple(v['phase'] for v in p['phases']) for p in reports}=={('parent',),('child',)}


def test_planner_main_profiling_preserves_model_setup_order(tmp_path):
    source=ROOT/'mcp_servers/worldengine/core/projects/AlgEngine/closed_loop/sim_test.py'
    main=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    class Config(dict):
        def __getattr__(self,key):return self[key]
        __setattr__=dict.__setitem__
    cfg=Config(model=SimpleNamespace(pretrained=None,train_cfg=None),data=SimpleNamespace(test=Config()),sim={})
    calls=[]
    class Model:
        def cuda(self):calls.append('cuda');return self
        def eval(self):calls.append('eval')
    async def inference(model,cfg,logger):calls.append('inference')
    def build(*args,**kwargs):calls.append('build');return Model()
    def checkpoint(*args,**kwargs):calls.append('checkpoint')
    def parallel(model,**kwargs):calls.append('parallel');return model
    timer=StartupTiming('planner',tmp_path)
    namespace=dict(_STARTUP=timer,_startup_span=timer.span,asyncio=asyncio,logging=logging,
        parse_args=lambda:SimpleNamespace(config='fixture',checkpoint='offline.pth',log_dir='',cfg_options=None,seed=None,fuse_conv_bn=False),
        Config=SimpleNamespace(fromfile=lambda path:cfg),get_root_logger=lambda **kwargs:SimpleNamespace(info=lambda *args:None),
        build_model=build,load_checkpoint=checkpoint,MMDataParallel=parallel,run_inference_loop=inference)
    exec(compile(ast.Module(body=[main],type_ignores=[]),str(source),'exec'),namespace)
    namespace['main']()
    assert calls==['build','checkpoint','cuda','parallel','eval','inference']
    phases=json.loads(next(tmp_path.glob('planner-*.json')).read_text())['phases']
    assert {'configuration_load','model_build','checkpoint_load_cpu','cuda_model_setup','model_ready'} <= {p['phase'] for p in phases}
    assert all(p['status']=='completed' for p in phases)
