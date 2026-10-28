#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$PROJECT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
用法：
  ./start.sh                       进入交互式任务入口
  ./start.sh "你的场景生成需求"     执行一次任务
  ./start.sh --check               检查配置和运行环境，不调用模型
  ./start.sh --init-local          从同级原项目复用本机环境、密钥文件和检索库
  ./start.sh --max-steps 40 "需求"  为本次任务设置步数上限
  ./start.sh --reconstruction-backend ufo "需求"  UFO 重建和编辑（默认）
  ./start.sh --reconstruction-backend dggt "需求" 使用原 DGGT 链路
HELP
  exit 0
fi

if [[ "${1:-}" == "--init-local" ]]; then
  # Only reuse local runtime resources; project source remains independent.
  exec python3 -B -m agent.local_runtime
fi

AGENT_PYTHON="$PROJECT_DIR/environments/agent/.venv/bin/python"
if [[ ! -x "$AGENT_PYTHON" ]]; then
  printf '%s\n' 'Agent 环境尚未准备。当前服务器可先执行：./start.sh --init-local' >&2
  printf '%s\n' '其他机器请按 README.md 完成首次环境安装。' >&2
  exit 1
fi
exec "$AGENT_PYTHON" -u -m agent.cli "$@"
