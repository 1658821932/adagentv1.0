import hashlib
import json
import os
import tempfile
import unittest
import builtins
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from agent.config import AgentConfig
from agent.executor import ToolExecutor
from mcp_servers.ufo.lane_config import generate


class BackendRoutingTests(unittest.TestCase):
    def test_default_and_explicit_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(AgentConfig.load().reconstruction_backend, 'ufo')
        with patch.dict(os.environ, {'RECONSTRUCTION_BACKEND': 'dggt'}):
            self.assertEqual(AgentConfig.load().reconstruction_backend, 'dggt')
        with patch.dict(os.environ, {'RECONSTRUCTION_BACKEND': 'bad'}):
            with self.assertRaises(ValueError):
                AgentConfig.load()

    def test_ufo_pipeline_pins_backend_and_passes_successful_asset(self):
        executor = ToolExecutor({'reconstruction': 'real', 'instance_discovery': 'real', 'scene_editor': 'real'})
        reconstruction = {'backend': 'ufo_worldengine_time_v2', 'workspace': '/test/scene', 'frame_count': 40}
        with patch('agent.executor.call_tool') as call:
            call.return_value = {'status': 'success', 'tool': 'reconstruction', 'result': reconstruction}
            self.assertEqual(executor.execute('reconstruction', {'backend': 'dggt'})['status'], 'success')
            self.assertEqual(call.call_args.args[1]['backend'], 'ufo')
            call.return_value = {'status': 'success', 'tool': 'instance_discovery', 'result': {}}
            executor.execute('instance_discovery', {'backend': 'dggt_yolo'})
            self.assertEqual(call.call_args.args[1]['ufo_reconstruction'], reconstruction)
            self.assertEqual(call.call_args.args[1]['backend'], 'ufo')
            call.return_value = {'status': 'success', 'tool': 'scene_editor', 'result': {}}
            executor.execute('scene_editor', {'edit_config_path': '/edit.json', 'target_selection_path': '/selected.json'})
            self.assertEqual(call.call_args.args[1]['ufo_reconstruction'], reconstruction)
            self.assertEqual(call.call_args.args[1]['operation'], 'lane_change')
            call.reset_mock()
            self.assertEqual(executor.execute('scene_editor', {})['status'], 'error')
            call.assert_not_called()
            executor.reset()
            self.assertEqual(executor.execute('instance_discovery', {})['status'], 'error')

    def test_dggt_pipeline_remains_separate(self):
        executor = ToolExecutor(reconstruction_backend='dggt')
        self.assertEqual(executor.prepare_arguments('instance_discovery', {})['backend'], 'dggt_yolo')
        self.assertEqual(executor.prepare_arguments('scene_editor', {'backend': 'ufo'})['backend'], 'dggt')
        self.assertNotIn('ufo_reconstruction', executor.prepare_arguments('scene_editor', {}))

    def test_generated_config_requires_selected_binding_and_explicit_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            candidate = root / 'candidates.json'
            selection, instance_path = root / 'selection.json', root / 'selected.json'
            selected = {'instance_id': 0, 'track_id': 10001, 'sam_track_id': 10001,
                        'frames': [{'frame_index': 0, 'bbox': [1, 2, 3, 4]}]}
            candidate.write_text(json.dumps({'instances': [selected], 'selection_required': True}))
            selection.write_text(json.dumps({'candidate_info_path': str(candidate),
                'candidate_info_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
                'selected_instance_info_path': str(instance_path), 'decision': 'selected',
                'selected_instance_id': 0, 'confidence': .9, 'reason': 'white car',
                'ranking': [{'instance_id': 0, 'reason': 'white car'}],
                'evidence': [{'frame_index': 0, 'observation': 'white car'}]}))
            instance_path.write_text(json.dumps({'instances': [selected], 'target_selection_path': str(selection)}))
            args = {'instance_info_path': str(instance_path), 'output_path': str(root / 'edit.json'),
                    'edit_plan': 'selected white car changes right', 'direction': 'right'}
            result = generate(args)
            self.assertEqual(result['direction'], 'right')
            self.assertFalse(result['metric_calibrated'])
            from tools.tool08_edit_config_generator.real import run as config_tool
            self.assertEqual(config_tool({**args, 'backend': 'ufo'})['result']['saved_path'], str((root / 'edit.json').resolve()))
            from tools.tool09_scene_editor.real import run as editor_tool
            with patch('mcp_servers.ufo.adapter.edit_lane_change', return_value={'rendered_video': '/edited.mp4'}) as editor:
                editor_tool({'backend': 'ufo', 'edit_config_path': str(root / 'edit.json'),
                             'target_selection_path': str(selection), 'ufo_reconstruction': {'workspace': '/test'}})
                self.assertEqual(editor.call_args.kwargs['direction'], 'right')
                self.assertEqual(editor.call_args.kwargs['target_selection_path'], str(selection))
                with self.assertRaisesRegex(ValueError, 'differs'):
                    editor_tool({'backend': 'ufo', 'edit_config_path': str(root / 'edit.json'),
                                 'target_selection_path': str(root / 'wrong.json')})
            with self.assertRaisesRegex(ValueError, 'explicit direction'):
                generate({**args, 'direction': None})
            candidate.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                generate(args)

    def test_cli_backend_overrides_without_running_agent(self):
        from agent import cli
        with patch.dict(os.environ, {'RECONSTRUCTION_BACKEND': 'ufo'}), \
             patch.object(cli, 'load_settings', return_value={}), \
             patch.object(cli, 'check_runtime', return_value=0) as check:
            self.assertEqual(cli.main(['--reconstruction-backend', 'dggt', '--check']), 0)
            check.assert_called_once()
            self.assertEqual(os.environ['RECONSTRUCTION_BACKEND'], 'dggt')

    def test_agent_without_pillow_delegates_ufo_evidence_to_worker(self):
        from tools.tool05_instance_discovery.grounded import anchored_frame_card
        actual_import = builtins.__import__

        def without_pillow(name, *args, **kwargs):
            if name == 'PIL':
                raise ImportError('Agent has no imaging dependencies')
            return actual_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / 'candidates.json'
            source.write_text(json.dumps({'source': {'backend': 'ufo'}}))
            card = {'path': str(root / 'card.jpg'), 'sha256': 'abc', 'frame_indices': [8]}
            with patch('builtins.__import__', side_effect=without_pillow), \
                 patch('mcp_servers.ufo.runtime.python_path', return_value=Path('/ufo/python')), \
                 patch('subprocess.run', return_value=SimpleNamespace(stdout=json.dumps(card))) as worker:
                self.assertEqual(anchored_frame_card(source, {'instance_id': 0}, 8, root / 'card.jpg'), card)
                self.assertEqual(worker.call_args.args[0][:3], ['/ufo/python', '-m', 'mcp_servers.ufo.evidence'])


if __name__ == '__main__':
    unittest.main()
