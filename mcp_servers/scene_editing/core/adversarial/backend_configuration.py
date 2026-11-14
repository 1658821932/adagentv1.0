"""Resolve bundled backend paths against the project, independent of child cwd."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PATH_FIELDS = {
    'vadv2': ('python', 'worldengine_root', 'config_path', 'checkpoint_path'),
    'pdm': ('python', 'worldengine', 'system_lib', 'map_root'),
    'render': ('python', 'worldengine_dir', 'system_lib', 'include_root', 'gsplat_extension'),
}


def load_settings(path=None, *, project_root=None):
    settings = json.loads(Path(path or Path(__file__).with_name('backend_settings.json')).read_text())
    root = Path(project_root or PROJECT_ROOT)
    for section, fields in PATH_FIELDS.items():
        for name in fields:
            value = settings.get(section, {}).get(name)
            if value is None or value == '':
                continue
            if not isinstance(value, str):
                raise ValueError(f'{section}.{name} must be a path string')
            target = Path(value).expanduser()
            # Resolving interpreter symlinks would bypass the virtual environment.
            settings[section][name] = os.path.abspath(target if target.is_absolute() else root / target)
    return settings
