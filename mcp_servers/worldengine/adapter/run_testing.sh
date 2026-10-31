#!/usr/bin/env bash
set -euo pipefail
ADAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ADAPTER_DIR/../../.." && pwd)"
PYTHON="${WORLDENGINE_LAUNCH_PYTHON:-$PROJECT_ROOT/environments/agent/.venv/bin/python}"
cd "$PROJECT_ROOT"
exec "$PYTHON" -m mcp_servers.worldengine.adapter.launch "$@"
