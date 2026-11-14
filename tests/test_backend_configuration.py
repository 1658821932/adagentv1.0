"""Backend launch paths must survive worktree relocation and different child cwd."""
import json
from pathlib import Path

import pytest

from mcp_servers.scene_editing.core.adversarial.backend_configuration import load_settings, PATH_FIELDS


def test_all_backend_consumers_get_absolute_paths_without_losing_venv(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    python = root / 'environments/scene_render/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to('/usr/bin/python3')
    original = {section: {field: 'environments/scene_render/bin/python' if field == 'python'
                         else f'resources/{field}' for field in fields}
                for section, fields in PATH_FIELDS.items()}
    original['defaults'] = dict(defense_initial_foci_enabled=True, attack_max_turns=12)
    original['render']['gsplat_extension'] = None
    original['render']['system_lib'] = '/usr/lib/x86_64-linux-gnu'
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    settings = load_settings(path, project_root=root)
    for section, fields in PATH_FIELDS.items():
        for name in fields:
            if original[section][name] is not None:
                assert Path(settings[section][name]).is_absolute()
                if name == 'python':
                    assert settings[section][name] == str(python)
    assert settings['render']['system_lib'] == '/usr/lib/x86_64-linux-gnu'
    assert settings['render']['gsplat_extension'] is None
    assert settings['defaults'] == original['defaults']
    assert path.read_bytes() == before


def test_absolute_local_settings_keep_their_effective_values():
    path = Path(__file__).resolve().parents[1] / 'mcp_servers/scene_editing/core/adversarial/backend_settings.json'
    original = json.loads(path.read_text())
    settings = load_settings(path)
    for section, fields in PATH_FIELDS.items():
        for name in fields:
            value = original.get(section, {}).get(name)
            if value and Path(value).is_absolute():
                assert settings[section][name] == value
    assert settings['defaults'] == original['defaults']


def test_invalid_path_type_names_the_bad_field(tmp_path):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(dict(render=dict(python=['invalid']))))
    with pytest.raises(ValueError, match=r'render\.python'):
        load_settings(path)
