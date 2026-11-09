"""Run a renderer with an explicitly supplied existing gsplat CUDA extension."""
import argparse
import importlib.util
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extension = args.extension.resolve(strict=True)
    script = args.script.resolve(strict=True)
    if extension.name != 'gsplat_cuda.so':
        raise ValueError('Expected an explicitly configured gsplat_cuda.so')
    import torch
    import gsplat
    spec = importlib.util.spec_from_file_location('gsplat_cuda', extension)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    gsplat.csrc = backend
    print('Render runtime: existing gsplat CUDA extension loaded; gsplat', gsplat.__version__,
          'torch', torch.__version__, flush=True)
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
