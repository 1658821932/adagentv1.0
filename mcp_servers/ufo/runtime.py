"""Pinned local resources and environment for offline UFO subprocesses."""
from __future__ import annotations

import os
from pathlib import Path

YX = Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx')
REMOTE_PROJECT = Path(os.environ.get('UFO_PROJECT_ROOT',
    '/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0')).resolve()
PROJECT = Path(__file__).resolve().parents[2]


def resources() -> dict[str, Path]:
    entry = Path(os.environ.get('UFO_ENTRY_ROOT', str(YX / 'ufo_video_entry_20261012'))).resolve()
    return {
        'entry': entry,
        'legacy': entry / 'releases/legacy_before_coherent_v1/code',
        'install': YX / 'exufo_singleview_delivery_20261009',
        'custom': YX / 'exufo_custom_rgb_20261008_210945',
        'source': Path(os.environ.get('UFO_MODEL_SOURCE', '/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo/e2_front352_20260930/source')).resolve(),
        'worldengine': Path(os.environ.get('UFO_WORLDENGINE_ROOT', str(YX / 'WorldEngine'))).resolve(),
        'env': Path(os.environ.get('UFO_ENV_ROOT', str(REMOTE_PROJECT / 'environments/ufoex'))).resolve(),
        'offline_deps': YX / 'moge_depth_full_20260922/source/offline_deps',
    }


def offline_env(gpu: int = 0) -> dict[str, str]:
    r = resources()
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    paths = [PROJECT / 'mcp_servers/ufo/offline_bootstrap', PROJECT,
             r['offline_deps'], r['install'] / 'frontend',
             r['install'] / 'python_deps', r['install'] / 'grounded_sam']
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, paths)), PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', UFO_OFFLINE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        HF_HOME=str(r['install'] / 'cache/huggingface'),
        TORCH_HOME=str(r['install'] / 'cache/torch_home'),
        CUDA_VISIBLE_DEVICES=str(gpu), CUDA_HOME='/usr/local/cuda',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MAX_JOBS='2',
        PATH=os.pathsep.join([str(r['env'] / 'bin'), '/usr/local/cuda/bin', env.get('PATH', '')]),
        LD_LIBRARY_PATH=os.pathsep.join([str(r['env'] / 'lib'), env.get('LD_LIBRARY_PATH', '')]).rstrip(os.pathsep),
        TORCH_EXTENSIONS_DIR=str(r['install'] / 'cache/torch_extensions'),
    )
    return env


def python_path() -> Path:
    path = resources()['env'] / 'bin/python'
    if not path.is_file():
        raise FileNotFoundError(f'UFO interpreter missing: {path}')
    return path
