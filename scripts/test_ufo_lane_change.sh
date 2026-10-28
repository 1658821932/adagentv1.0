#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UFO_PROJECT_ROOT=${UFO_PROJECT_ROOT:-/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagentv1.0}
export UFO_ENV_ROOT=${UFO_ENV_ROOT:-$UFO_PROJECT_ROOT/environments/ufoex}
export UFO_ENTRY_ROOT=${UFO_ENTRY_ROOT:-/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/ufo_video_entry_20261012}
export UFO_WORLDENGINE_ROOT=${UFO_WORLDENGINE_ROOT:-/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/WorldEngine}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PATH="$UFO_ENV_ROOT/bin:$PATH"
export PYTHONPATH="$CODE_ROOT"
unset PYTHONHOME
if [[ ! -x "$UFO_ENV_ROOT/bin/python" ]]; then
  echo "UFO Python missing: $UFO_ENV_ROOT/bin/python" >&2
  exit 1
fi
cd "$CODE_ROOT"
exec "$UFO_ENV_ROOT/bin/python" -m mcp_servers.ufo.smoke "$@"
