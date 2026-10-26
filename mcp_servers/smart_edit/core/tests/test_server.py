"""通过真实 stdio 子进程检查 MCP 资源接口,无需模型服务。"""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ServerResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_resource_catalogs_match_asset_tools(self):
        project = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix="smart-edit-test-") as output_dir:
            env = os.environ.copy()
            env.update(
                ASSETS_DIR=str(project / "assets"),
                SMPL_ASSETS_DIR=str(project / "assets/pedestrian"),
                OUTPUT_DIR=output_dir,
            )
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "smart_edit"],
                cwd=str(project),
                env=env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resources = (await session.list_resources()).resources
                    expected = {"assets://catalog": "list_assets"}
                    self.assertEqual(
                        {str(resource.uri) for resource in resources}, set(expected)
                    )
                    for resource in resources:
                        with self.subTest(uri=str(resource.uri)):
                            catalog = await session.read_resource(resource.uri)
                            tool = await session.call_tool(expected[str(resource.uri)], {})
                            self.assertFalse(tool.isError)
                            self.assertEqual(
                                json.loads(catalog.contents[0].text),
                                json.loads(tool.content[0].text),
                            )


if __name__ == "__main__":
    unittest.main()
