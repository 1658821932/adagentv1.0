"""Project-level CLI; resolve configuration before importing tool modules."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_settings() -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    path = ROOT / 'configs/system.local.json'
    if not path.is_file():
        return {}
    settings = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(settings, dict):
        raise ValueError('configs/system.local.json 必须为 JSON 对象')
    context = settings.get('task_context', {})
    if not isinstance(context, dict):
        raise ValueError('task_context 必须为 JSON 对象')
    return settings


def build_task(task: str, settings: dict) -> str:
    context = settings.get('task_context', {})
    if not context:
        return task
    return task + '\n\n系统预配置资源（按需使用；未配置的资源不得猜测）：\n' + json.dumps(context, ensure_ascii=False, indent=2)


def check_runtime() -> int:
    from agent.config import AgentConfig
    config = AgentConfig.load()
    missing = []
    for primary, alias in (('OPENAI_API_KEY', 'LLM_API_KEY'),):
        if not (os.environ.get(primary) or os.environ.get(alias)):
            missing.append(primary)
    real = {name for name, mode in config.tool_modes.items() if mode == 'real'}
    prefixes = set()
    if config.reconstruction_backend == 'ufo' and real & {'reconstruction', 'scene_editor', 'instance_discovery'}:
        from mcp_servers.ufo.runtime import python_path
        prefixes.add(('UFO Python', python_path()))
    if config.reconstruction_backend == 'dggt' and real & {'reconstruction', 'scene_editor', 'instance_discovery'}:
        prefixes.add(('DGGT Python', Path(os.environ.get('DGGT_PYTHON') or ROOT / 'environments/dggt/.conda/bin/python')))
    if config.reconstruction_backend == 'dggt' and real & {'reconstruction', 'scene_editor'}:
        prefixes.add(('SegFormer Python', Path(os.environ.get('DGGT_SEGFORMER_PYTHON') or ROOT / 'environments/dggt_segformer/.conda/bin/python')))
    for tool, prefix in (('video_retriever', 'video_retriever/.venv'),
                         ('edit_config_generator', 'smart-edit/.venv'),
                         ('policy_runner', 'algengine/.conda'),
                         ('policy_runner', 'simengine/.conda'),
                         ('failure_case_filter', 'simengine/.conda'),
                         ('policy_trainer', 'algengine/.conda')):
        if tool in real and not (tool == 'edit_config_generator' and config.reconstruction_backend == 'ufo'):
            prefixes.add((prefix, ROOT / 'environments' / prefix / 'bin/python'))
    for label, path in sorted(prefixes):
        if not path.expanduser().is_file():
            missing.append(f'{label}: {path}')
    print(f'项目：{ROOT}\nAgent 配置：正常；最多 {config.max_steps} 步；重建后端：{config.reconstruction_backend}')
    if missing:
        print('缺少配置或环境：\n- ' + '\n- '.join(missing))
        return 1
    print('基础配置和 Python 入口检查通过；未调用模型、检查索引内容或执行 GPU 任务。')
    return 0


def run_task(task: str, settings: dict, max_steps: int | None) -> int:
    from agent.main import Agent
    agent = Agent()
    output_dir = ROOT / 'outputs/tasks' / datetime.now().strftime('task_%Y%m%d_%H%M%S_%f')
    output_dir.mkdir(parents=True, exist_ok=False)
    context = {'video_library': 'default', **settings.get('task_context', {}),
               'project_root': str(ROOT), 'output_dir': str(output_dir),
               'reconstruction_backend': agent.config.reconstruction_backend}
    print(f'任务输出目录：{output_dir}')
    result = agent.run(build_task(task, {'task_context': context}), max_steps=max_steps)
    print(f"\n任务状态：{result.get('status', 'unknown')}")
    if agent.logger:
        print(f'运行记录：{agent.logger.run_dir}')
    if result.get('error'):
        print(f"原因：{result['error']}", file=sys.stderr)
    return 0 if result.get('status') == 'finished' else 1


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('步数必须为正整数')
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='自动驾驶场景生成系统')
    parser.add_argument('task', nargs='*', help='自然语言任务；省略时交互输入')
    parser.add_argument('--check', action='store_true', help='仅检查配置和环境入口')
    parser.add_argument('--max-steps', type=positive_int, default=None)
    parser.add_argument('--reconstruction-backend', choices=['ufo', 'dggt'], default=None)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    try:
        settings = load_settings()
        if args.reconstruction_backend:
            os.environ['RECONSTRUCTION_BACKEND'] = args.reconstruction_backend
        if args.check:
            return check_runtime()
        if args.task:
            task = ' '.join(args.task).strip()
            if not task:
                parser.error('任务不能为空')
            return run_task(task, settings, args.max_steps)
        if not sys.stdin.isatty():
            parser.error('非交互运行时请提供任务，例如 ./start.sh "生成前车切入场景"')
        print('自动驾驶场景生成系统\n输入自然语言需求；输入 exit 退出。每个任务独立执行。')
        while True:
            task = input('\nadagent> ').strip()
            if task.lower() in {'exit', 'quit'}:
                return 0
            if not task:
                continue
            try:
                run_task(task, settings, args.max_steps)
            except Exception as exc:
                print(f'任务失败：{exc}', file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print('\n已退出。')
        return 130
    except Exception as exc:
        print(f'启动失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
