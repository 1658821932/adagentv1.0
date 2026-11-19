"""Snapshots and non-overlapping timing views for the main Agent run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from common.llm_audit import Sanitizer, utc
from common.trace_io import write_json

EDIT_TOOLS = {
    'construct_attack_candidates', 'construct_trajectory', 'construct_defense_candidates',
    'edit_trajectory_points', 'edit_speed_profile', 'edit_route', 'edit_motion',
    'repair_candidate', 'optimize_defense_candidate', 'edit_trajectory',
}


def snapshot(root, project):
    root, project = Path(root), Path(project)
    sanitizer = Sanitizer(root)
    def git(*args):
        result = subprocess.run(['git', *args], cwd=project, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    sources = {}
    # No weights, datasets, runtime outputs, vendor trees, or .env files.
    directories = ['agent', 'common', 'tools', 'configs',
                   'mcp_servers/scene_editing/adapter', 'mcp_servers/scene_editing/core/adversarial',
                   'mcp_servers/worldengine/adapter', 'mcp_servers/worldengine_failure_analysis',
                   'mcp_servers/sentry_video_search/adapter', 'mcp_servers/sentry_video_search/core']
    excluded = {'runtime', '__pycache__', 'experiments', 'data', 'vendor', '.venv', '.git', 'node_modules'}
    paths = {project/'api_config.py', project/'mcp_servers/runtime.py', project/'mcp_servers/process.py',
             project/'mcp_servers/sentry_video_search/core/vendor/sentrysearch/qwen_api_embedder.py'}
    for directory in directories:
        for parent, children, files in os.walk(project/directory):
            children[:] = [name for name in children if name not in excluded and not name.startswith('.')]
            for name in files:
                path = Path(parent)/name
                if path.suffix in {'.py', '.md', '.json'}:
                    paths.add(path)
    for path in sorted(paths):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        relative = str(path.relative_to(project))
        sources[relative] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        if path.suffix in {'.md', '.json'}:
            value = raw.decode('utf-8')
            destination = root/'snapshot'/'files'/relative
            if path.suffix == '.json':
                write_json(destination, sanitizer.clean(json.loads(value)))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(sanitizer.text(value), encoding='utf-8')
    write_json(root/'snapshot'/'manifest.json', sanitizer.clean(dict(
        created_at=utc(), git_head=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
        sources=sources, note='Prompt/config contents are copied with redaction; source code is identified by SHA256, not copied.')))


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def summarize(root):
    root = Path(root)
    run = read(root/'run.json') or {}
    requests = [r for path in (root/'api_requests').glob('*/result.json') if (r := read(path))]
    stages = {}
    def stage(name):
        return stages.setdefault(name, dict(stage=name, tool_calls=0, tool_wall_seconds=0.,
            api_calls=0, api_seconds=0., api_errors=0, api_incomplete=0,
            prompt_tokens=0, completion_tokens=0, usage_available_calls=0,
            max_prompt_tokens=None, max_input_text_characters=0, max_image_count=0, max_tool_schema_bytes=0,
            subagent_tool_calls=0, edit_attempts=0, retry_requests=0, main_correction_requests=0))
    for request in requests:
        row = stage(request['stage'])
        row['api_calls'] += 1
        row['api_seconds'] += request.get('elapsed_seconds', 0)
        row['api_errors'] += request['status'] in {'error', 'interrupted'}
        row['api_incomplete'] += request['status'] == 'running'
        row['retry_requests'] += request.get('attempt_index', 1) > 1
        row['main_correction_requests'] += int(request.get('context', {}).get('decision_attempt', 1)) > 1
        usage = request.get('usage')
        metrics = request.get('input_metrics') or {}
        for name, key in [('max_input_text_characters', 'text_characters'), ('max_image_count', 'image_count'),
                          ('max_tool_schema_bytes', 'tool_schema_bytes')]:
            row[name] = max(row[name], metrics.get(key, 0))
        if isinstance(usage, dict):
            row['usage_available_calls'] += 1
            row['prompt_tokens'] += usage.get('prompt_tokens') or usage.get('input_tokens') or 0
            row['completion_tokens'] += usage.get('completion_tokens') or usage.get('output_tokens') or 0
            prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens'))
            if isinstance(prompt_tokens, (int, float)):
                row['max_prompt_tokens'] = max(row['max_prompt_tokens'] or 0, prompt_tokens)
    for folder in sorted((root/'tools').glob('step_*')):
        call, timing = read(folder/'call.json'), read(folder/'timing.json')
        if call:
            row = stage(call['tool'])
            row['tool_calls'] += 1
            row['tool_wall_seconds'] += (timing or {}).get('elapsed_seconds', 0) or 0
    for path in (root/'tool_events').glob('*.json'):
        event = read(path)
        if not event:
            continue
        context = event.get('context', {})
        tool_id = context.get('tool_call_id')
        row = stage(tool_id.rsplit(':', 1)[-1] if tool_id else context.get('stage', 'unknown'))
        row['subagent_tool_calls'] += 1
        row['edit_attempts'] += event.get('tool_name') in EDIT_TOOLS
    for row in stages.values():
        for key in ('tool_wall_seconds', 'api_seconds'):
            row[key] = round(row[key], 3)
    wall = None
    if run.get('start_time'):
        end = datetime.fromisoformat(run['end_time']) if run.get('end_time') else datetime.now(timezone.utc)
        wall = round((end-datetime.fromisoformat(run['start_time'])).total_seconds(), 3)
    report = dict(schema_version='run_summary.v1', updated_at=utc(), run_id=run.get('run_id'),
        status=run.get('status'), wall_seconds=wall,
        api_seconds=round(sum(r['api_seconds'] for r in stages.values()), 3),
        api_calls=len(requests), stages=list(stages.values()),
        notes=['API time includes all recorded attempts, including errors and main corrections.',
               'API sums can exceed wall time for parallel requests; API time is contained in tool wall time.',
               'Running/interrupted process requests without a final record are incomplete, not zero-cost successes.',
               'Edit attempts count archived editing tool invocations, including rejected/invalid attempts and previews.',
               'Token totals cover provider-reported usage only; missing usage is not estimated.'])
    write_json(root/'summary.json', report)
    lines = [f"# 运行报告 {report['run_id']}", '',
             f"状态：{report['status']}；运行总耗时：{wall} 秒；累计 API：{report['api_seconds']} 秒 / {len(requests)} 次。", '',
             '| 模块 | 工具总秒数 | API秒数 | API次数 | 重试/纠错 | API错误 | 未完成请求 | 编辑尝试 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in stages.values():
        lines.append(f"| {row['stage']} | {row['tool_wall_seconds']} | {row['api_seconds']} | {row['api_calls']} | {row['retry_requests']} | {row['api_errors']} | {row['api_incomplete']} | {row['edit_attempts']} |")
    lines += ['', '| 模块 | 最大输入 Token | 最大输入文本字符数 | 最大图片数 | 最大工具定义字节数 | usage覆盖 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in stages.values():
        tokens = row['max_prompt_tokens'] if row['max_prompt_tokens'] is not None else '未返回'
        lines.append(f"| {row['stage']} | {tokens} | {row['max_input_text_characters']} | {row['max_image_count']} | {row['max_tool_schema_bytes']} | {row['usage_available_calls']}/{row['api_calls']} |")
    lines += ['', 'API 时间包含在工具总时间中，不能相加；并发请求的累计 API 时间可能超过运行总耗时。',
              '未返回 usage 的请求不估算 Token。编辑尝试包含预览、被拒绝及参数校验失败的调用。',
              '请求上下文见 api_requests/；图片见 assets/；子工具输入输出见 tool_events/；内部阶段计时见 operations/；提示词和配置快照见 snapshot/。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    import sys
    summarize(Path(sys.argv[1]))
