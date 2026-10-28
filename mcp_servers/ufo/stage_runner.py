"""Execute a reference stage with explicit workspace and model source paths."""
from __future__ import annotations
import os
import runpy
import sys
import types
from pathlib import Path
from .runtime import resources


def main():
    r = resources()
    script = Path(sys.argv[1]).resolve()
    module = types.ModuleType('runtime_paths')
    module.WORK = Path(os.environ['UFO_WORKSPACE']).resolve()
    module.INSTALL, module.LEGACY = r['install'], r['custom']
    module.MODEL_CONFIG = r['custom'] / 'run_config.json'
    module.MODEL_SOURCE = r['source']
    module.PRED = module.WORK / 'reconstruction/000/pred100k'
    module.EDIT, module.BACK = module.WORK / 'editing/000', module.WORK / 'background/000'
    sys.modules['runtime_paths'] = module
    sys.path[:0] = [str(script.parent), str(r['legacy'])]
    if script.name in ('build_scene100.py', 'export_ufo_background.py'):
        from .model_loading import install_for_source
        install_for_source(r['source'])
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
