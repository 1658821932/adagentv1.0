from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
def configured_path(name, env_name, default, *, follow_symlinks=True):
    config_path=PROJECT_ROOT/'configs/nuplan_paths.local.json'
    local=json.loads(config_path.read_text()) if config_path.is_file() else {}
    if not isinstance(local,dict):
        raise ValueError('nuplan_paths.local.json must contain a JSON object')
    value=os.environ.get(env_name) or local.get(name) or str(default)
    path=Path(value).expanduser()
    path=path if path.is_absolute() else PROJECT_ROOT/path
    return path.resolve() if follow_symlinks else Path(os.path.abspath(path))


NEXUS_ROOT = configured_path('adscenegraph_parent','ADSCENEGRAPH_PARENT',PROJECT_ROOT.parent/'nexus_repro/Nexus')
ADSCENEGRAPH_ROOT = NEXUS_ROOT/'Omniscenegraph'
RECONSTRUCTION_PYTHON = configured_path('reconstruction_python','ADSCENEGRAPH_PYTHON',NEXUS_ROOT/'artifacts/nexus_3dgs/venv/bin/python',follow_symlinks=False)
DATA_ROOT = PROJECT_ROOT/'mcp_servers/sentry_video_search/data_nuplan'
ARTIFACT_ROOT = PROJECT_ROOT/'artifacts/nuplan_retrieval'
RAW_ROOT = configured_path('raw_root','NUPLAN_RAW_ROOT',PROJECT_ROOT/'data/nuplan')


def nuplan_reader():
    """Reuse the existing, read-only nuPlan geometry/calibration implementation."""
    if str(NEXUS_ROOT) not in sys.path:
        sys.path.insert(0, str(NEXUS_ROOT))
    from nexus_3dgs import nuplan
    return nuplan


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def require_local_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f'Output must remain inside the independent adagent copy: {path}')
    return path


def runtime_environment():
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WANDB_MODE='disabled')
    lib=PROJECT_ROOT/'runtime_libs/usr/lib/x86_64-linux-gnu'
    if lib.is_dir():
        old=env.get('LD_LIBRARY_PATH','')
        env['LD_LIBRARY_PATH']=str(lib)+(':'+old if old else '')
    env['MPLCONFIGDIR']=str(ARTIFACT_ROOT/'cache/matplotlib')
    env['PYTHONPATH']=str(ADSCENEGRAPH_ROOT)+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    env.setdefault('MAX_JOBS','4')
    env['TORCH_EXTENSIONS_DIR']=str(ARTIFACT_ROOT/'cache/torch_extensions')
    env['TORCH_HOME']=str(ARTIFACT_ROOT/'cache/torch')
    return env
