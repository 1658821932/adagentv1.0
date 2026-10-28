"""Explicit, non-overwriting reuse of an existing sibling deployment."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_missing(source: Path, destination: Path) -> str:
    if destination.is_symlink():
        return '保留已有'
    if destination.is_dir() and source.is_dir():
        for child in source.iterdir():
            if child.name != '.gitkeep':
                link_missing(child, destination / child.name)
        return '已合并缺少的资源（保留已有）'
    if destination.exists():
        return '保留已有'
    if not source.exists():
        return '源资源缺失'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return '已复用'


def main() -> int:
    source = ROOT.parent / 'adagentv1.0'
    if source.resolve() == ROOT.resolve() or not (source / 'agent/main.py').is_file():
        print('没有找到同级原部署 adagentv1.0，请按使用指南准备环境。')
        return 1
    resources = ['.env']
    resources += ['environments/' + name for name in (
        'agent/.venv', 'video_retriever/.venv', 'dggt/.conda',
        'dggt_segformer/.conda', 'smart-edit/.venv', 'algengine/.conda',
        'simengine/.conda', 'uniad/.conda', 'ufoex')]
    resources += ['mcp_servers/sentry_video_search/' + name for name in (
        'data', 'runtime', 'data_waymo_front')]
    print('复用原部署的运行资源；环境、密钥文件和检索库将通过符号链接共享。')
    for relative in resources:
        print(f'{link_missing(source / relative, ROOT / relative)}：{relative}')
    print('完成。执行 ./start.sh --check 检查，再执行 ./start.sh 输入任务。')
    return 0 if (ROOT / 'environments/agent/.venv/bin/python').is_file() else 1


if __name__ == '__main__':
    raise SystemExit(main())
