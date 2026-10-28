"""Adopt completed stages after adapter repairs without weakening input checks."""
from __future__ import annotations

import json
from pathlib import Path
import time


def stage_complete(marker: Path, expected: list[Path]) -> bool:
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
        return isinstance(record, dict) and all(path.is_file() and path.stat().st_size > 0 for path in expected)
    except (OSError, ValueError):
        return False


def check_resume_contract(previous: dict, current: dict, *, allow_adapter_updates: bool) -> list[str]:
    if previous == current:
        return []
    # Input, frame protocol, runtime, source path and checkpoint must remain exact.
    for key in set(previous) | set(current):
        if key != 'code' and previous.get(key) != current.get(key):
            raise ValueError(f'Cannot reuse results: {key} changed; use a new scene name')
    old_code, new_code = previous.get('code', {}), current.get('code', {})
    changed = sorted(path for path in set(old_code) | set(new_code) if old_code.get(path) != new_code.get(path))
    # A source/model/reference implementation change requires a new run. The
    # explicit resume flag only permits repairs within this adapter package.
    for path in changed:
        parts = Path(path).parts
        if not any(parts[i:i + 2] == ('mcp_servers', 'ufo') for i in range(len(parts) - 1)):
            raise ValueError(f'Cannot reuse results: external model/reference code changed: {path}')
    if changed and not allow_adapter_updates:
        raise ValueError('Adapter code changed; use --resume to reuse completed stages for the same input/model')
    return changed


def archive_contract(path: Path, previous: dict, changed: list[str]) -> None:
    history = path.parent / 'contract_history'
    history.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    (history / f'{stamp}.json').write_text(json.dumps({
        'previous_contract': previous, 'changed_adapter_files': changed,
        'reuse_policy': 'same input, frame protocol, environment, checkpoint and external source',
    }, ensure_ascii=False, indent=2) + '\n')
