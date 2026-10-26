"""终端客户端资源释放测试,只连接本地 MCP 服务。"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from clients.terminal_chat import SmartEditTerminalClient


class TerminalClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_closes_api_connection_when_session_body_fails(self):
        project = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix="smart-edit-client-test-") as output_dir:
            client = SmartEditTerminalClient(argparse.Namespace(
                api_key="test-key",
                base_url="http://127.0.0.1:1/v1",
                show_tool_process=False,
                assets_dir=str(project / "assets"),
                smpl_assets_dir=str(project / "assets/pedestrian"),
                output_dir=output_dir,
            ))
            with self.assertRaises(Exception) as caught:
                async with client:
                    self.assertTrue(client.tools)
                    raise RuntimeError("intentional client failure")
            self.assertIn("intentional client failure", repr(caught.exception))
            self.assertTrue(client.openai.is_closed())


if __name__ == "__main__":
    unittest.main()
