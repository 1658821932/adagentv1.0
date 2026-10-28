import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mcp_servers.ufo.offline_pipeline import preflight


class StartupTests(unittest.TestCase):
    def test_default_preflight_does_not_import_numerical_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            locations = {key: root / key for key in ('env', 'source', 'install', 'custom', 'legacy', 'worldengine')}
            locations['env'] = Path(sys.prefix).resolve()
            with patch('mcp_servers.ufo.offline_pipeline.resources', return_value=locations), \
                 patch('mcp_servers.ufo.offline_pipeline.python_path', return_value=Path(sys.executable)), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(sys, 'path', list(sys.path)), \
                 patch('mcp_servers.ufo.offline_pipeline.importlib.util.find_spec',
                       return_value=SimpleNamespace(origin='/local/package/__init__.py')), \
                 patch('mcp_servers.ufo.offline_pipeline.importlib.import_module',
                       side_effect=AssertionError('heavy import during lightweight preflight')):
                result = preflight(root / 'input.mp4')
            self.assertEqual(result['mode'], 'paths_only')
            self.assertEqual(result['imports'], {})
            self.assertIn('torch', result['module_locations'])
            self.assertFalse(result['gpu_inference_started'])


if __name__ == '__main__':
    unittest.main()
