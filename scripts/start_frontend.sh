#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
STATE_DIR="$FRONTEND_DIR/.cache"
STATE_FILE="$STATE_DIR/frontend-server.json"
PID_FILE="$STATE_DIR/frontend-server.pid"
LOG_FILE="$FRONTEND_DIR/frontend.log"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5173}"
URL="http://$HOST:$PORT"

if ! command -v node >/dev/null 2>&1; then
  echo "未找到 node，请先安装 Node.js。" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "未找到 curl，无法进行启动健康检查。" >&2
  exit 1
fi
if [[ ! -f "$FRONTEND_DIR/server.mjs" ]]; then
  echo "未找到前端服务入口：$FRONTEND_DIR/server.mjs" >&2
  exit 1
fi

healthy() {
  curl --silent --fail --max-time 3 --output /dev/null "$URL/" &&
    curl --silent --fail --max-time 5 --output /dev/null "$URL/api/runs?limit=1"
}

if healthy; then
  echo "前端已在运行：$URL"
  exit 0
fi

if curl --silent --max-time 3 --output /dev/null "$URL/"; then
  echo "$URL 已有服务响应，但运行列表接口未通过检查；不会重复启动，请检查端口占用。" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "此前启动的前端进程仍在（PID $OLD_PID），但健康检查未通过，请查看 $LOG_FILE。" >&2
    exit 1
  fi
fi

mkdir -p "$STATE_DIR"
NODE_BIN="$(command -v node)"
cd "$FRONTEND_DIR"
PID="$(NODE_BIN="$NODE_BIN" FRONTEND_DIR="$FRONTEND_DIR" PROJECT_ROOT="$PROJECT_ROOT" \
  LOG_FILE="$LOG_FILE" HOST="$HOST" PORT="$PORT" node <<'NODE'
const fs = require('node:fs');
const { spawn } = require('node:child_process');
const log = fs.openSync(process.env.LOG_FILE, 'a');
const child = spawn(process.env.NODE_BIN, ['server.mjs'], {
  cwd: process.env.FRONTEND_DIR,
  env: { ...process.env, ADAGENT_PROJECT_ROOT: process.env.PROJECT_ROOT },
  detached: true,
  stdio: ['ignore', log, log],
});
child.unref();
fs.closeSync(log);
process.stdout.write(String(child.pid));
NODE
)"
printf '%s\n' "$PID" >"$PID_FILE"

PID="$PID" NODE_BIN="$NODE_BIN" PROJECT_ROOT="$PROJECT_ROOT" FRONTEND_DIR="$FRONTEND_DIR" \
  STATE_FILE="$STATE_FILE" LOG_FILE="$LOG_FILE" URL="$URL" HOST="$HOST" PORT="$PORT" \
  node -e 'const fs = require("node:fs"); const data = {pid: Number(process.env.PID), host: process.env.HOST, port: Number(process.env.PORT), url: process.env.URL, cwd: process.env.FRONTEND_DIR, command: [process.env.NODE_BIN, process.env.FRONTEND_DIR + "/server.mjs"], started_at: Date.now() / 1000, log_path: process.env.LOG_FILE, reason: "scripts/start_frontend.sh"}; fs.writeFileSync(process.env.STATE_FILE, JSON.stringify(data, null, 2) + "\n");'

for ((attempt = 0; attempt < 60; attempt++)); do
  if healthy; then
    echo "前端启动成功：$URL"
    echo "PID：$PID"
    echo "日志：$LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "前端进程已退出，请查看日志：$LOG_FILE" >&2
    tail -n 20 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 0.5
done

echo "前端启动后 30 秒内未通过健康检查（PID $PID），请查看日志：$LOG_FILE" >&2
tail -n 20 "$LOG_FILE" >&2
exit 1
