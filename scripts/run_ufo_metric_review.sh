#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UFO_PROJECT_ROOT=${UFO_PROJECT_ROOT:-/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0}
export UFO_ENV_ROOT=${UFO_ENV_ROOT:-$UFO_PROJECT_ROOT/environments/ufoex}
export PYTHONPATH="$CODE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
cd "$CODE_ROOT"
exec "$UFO_ENV_ROOT/bin/python" -c '
import os,sys
from mcp_servers.ufo.runtime import python_path,offline_env
python=str(python_path())
os.execve(python,[python,"-u","-m","mcp_servers.ufo.review_run",*sys.argv[1:]],offline_env(int(os.environ.get("UFO_GPU","0"))))
' "$@"
