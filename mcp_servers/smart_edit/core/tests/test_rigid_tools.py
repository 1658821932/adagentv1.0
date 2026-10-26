"""刚体工具草稿完整性和连续编辑测试,无需模型 API。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


from smart_edit.nodes.rigid import RigidNodeBuilder
from smart_edit.tools.handlers import register_tool_handlers


class RigidToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        workspace = TemporaryDirectory(prefix="smart-edit-rigid-test-")
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        assets = self.root / "assets"
        self.car = assets / "car" / "taxi.ply"
        self.cone = assets / "cone" / "cone.ply"
        for path in (self.car, self.cone):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()  # 工具只验证目录中的路径,不读取点云内容。
        self.handlers = register_tool_handlers(str(assets), str(self.root / "outputs"))

    async def call(self, name, args):
        return (await self.handlers[name](args))[0].text

    async def preview(self):
        return json.loads(await self.call("preview_config", {}))

    async def add(self, **kwargs):
        args = {"asset": str(self.car), "anchor_pixel": [259, 250], "scale": 0.02}
        args.update(kwargs)
        return await self.call("add_rigid_asset", args)

    async def test_start_config_keeps_instance_info_path(self):
        instance_info = self.root / "instance_info.json"
        instance_payload = {"instances": [{
            "instance_id": 0,
            "frames": [
                {"frame_index": 7, "bbox": [75, 195, 175, 265]},
                {"frame_index": 8, "bbox": [78, 194, 178, 264]},
            ],
        }]}
        instance_info.write_text(json.dumps(instance_payload), encoding="utf-8")
        await self.call("start_config", {
            "instance_info_path": str(instance_info),
        })
        listed = json.loads(await self.call("list_instances", {
            "instance_info_path": str(instance_info),
        }))
        self.assertEqual(listed["instances"], instance_payload["instances"])
        self.assertEqual(listed["instance_info_path"], str(instance_info))
        await self.call("remove_rigid_instance", {"instance_id": 0})
        saved = json.loads(await self.call("save_config", {
            "output_path": "edit.json",
        }))
        config = json.loads(Path(saved["saved_path"]).read_text(encoding="utf-8"))
        self.assertEqual(config["instance_info_path"], str(instance_info))
        self.assertEqual(config, await self.preview())

    async def test_failed_add_does_not_change_draft(self):
        for invalid in (
            {"anchor_pixel": [259, 250], "scale": 0.02, "asset": ""},
            {"anchor_pixel": [259], "scale": 0.02},
            {"anchor_pixel": [259, 250], "scale": 0},
        ):
            with self.subTest(invalid=invalid):
                await self.call("reset_config", {})
                before = await self.preview()
                self.assertTrue((await self.add(**invalid)).startswith("参数错误:"))
                self.assertEqual(await self.preview(), before)
    async def test_empty_trajectory_does_not_change_draft(self):
        before = await self.preview()
        result = await self.call("edit_rigid_trajectory", {
            "instance_id": 0, "waypoints": [],
        })
        self.assertTrue(result.startswith("参数错误:"))
        self.assertEqual(await self.preview(), before)

    async def test_dggt_insertion_uses_anchor_pixel_and_scale(self):
        await self.add()
        node = (await self.preview())["Nodes"]["RigidNodes"]
        RigidNodeBuilder().validate(node)
        self.assertEqual(node["add"], {
            "add_obj": [str(self.car)],
            "anchor_pixel": [[259, 250]],
            "scale": [0.02],
        })

        await self.add(anchor_pixel=[300, 240], scale=0.03)
        node = (await self.preview())["Nodes"]["RigidNodes"]
        self.assertEqual(node["add"], {
            "add_obj": [str(self.car)],
            "anchor_pixel": [[300, 240]],
            "scale": [0.03],
        })


    async def test_vehicle_category_validation_preserves_draft(self):
        before = await self.preview()
        result = await self.call("add_rigid_asset", {
            "asset": str(self.cone),
            "anchor_pixel": [259, 250],
            "scale": 0.02,
        })
        self.assertTrue(result.startswith("参数错误:"))
        self.assertEqual(await self.preview(), before)
    async def test_multiple_edits_saved_as_valid_json(self):
        await self.call("start_config", {"instance_info_path": str(self.root / "instance_info.json")})
        for instance_id, waypoints in (
            (0, [[0, -2, 0]]),
            (1, [[0, 0, 0], [10, 2, 0]]),
            (2, [[0, 0, 0], [6, -1.5, 0], [12, -3, 0]]),
        ):
            await self.call("edit_rigid_trajectory", {
                "instance_id": instance_id, "waypoints": waypoints,
            })
        saved = json.loads(await self.call("save_config", {"output_path": "rigid.json"}))
        config = json.loads(Path(saved["saved_path"]).read_text(encoding="utf-8"))
        self.assertEqual(config, await self.preview())
        node = config["Nodes"]["RigidNodes"]
        RigidNodeBuilder().validate(node)
        self.assertEqual(node["trajectory_edit"]["instance_id"], [0, 1, 2])
        self.assertEqual(node["trajectory_edit"]["method"], ["linear", "linear", "cubic"])

    async def test_rotate_scale_and_replace_instances(self):
        await self.call("rotate_rigid_instance", {
            "instance_id": 0, "rotation": [0, 0, 30],
        })
        await self.call("scale_rigid_instance", {
            "instance_id": 1, "scale": 0.8,
        })
        await self.call("replace_rigid_instance", {
            "instance_id": 2, "asset": str(self.car), "scale": 0.02,
        })
        node = (await self.preview())["Nodes"]["RigidNodes"]
        RigidNodeBuilder().validate(node)
        self.assertEqual(node["rotate"], {
            "instance_id": [0], "rotation": [[0, 0, 30]],
        })
        self.assertEqual(node["scale"], {
            "instance_id": [1], "scale": [0.8],
        })
        self.assertEqual(node["replace"], {
            "target_id": [2], "replace_obj": [str(self.car)], "scale": [0.02],
        })

    async def test_remove_instances_deduplicates_and_clears_matching_edits(self):
        await self.call("start_config", {"instance_info_path": str(self.root / "instance_info.json")})
        for instance_id in (0, 1, 0):
            await self.call("edit_rigid_trajectory", {
                "instance_id": instance_id, "waypoints": [[0, -2, 0]],
            })
        await self.add()
        for instance_id in (0, 3, 0):
            result = await self.call("remove_rigid_instance", {"instance_id": instance_id})
            self.assertFalse(result.startswith("参数错误:"))
        saved = json.loads(await self.call("save_config", {"output_path": "remove.json"}))
        config = json.loads(Path(saved["saved_path"]).read_text(encoding="utf-8"))
        node = config["Nodes"]["RigidNodes"]
        RigidNodeBuilder().validate(node)
        self.assertEqual(node["remove"], {"instance_id": [0, 3]})
        self.assertEqual(node["trajectory_edit"], {
            "instance_id": [1], "waypoints": [[[0, -2, 0]]], "method": ["linear"],
        })
        self.assertEqual(node["add"]["anchor_pixel"], [[259, 250]])
        self.assertEqual(node["add"]["scale"], [0.02])
        await self.call("remove_rigid_instance", {"instance_id": 1})
        self.assertNotIn("trajectory_edit", (await self.preview())["Nodes"]["RigidNodes"])

    async def test_save_requires_dggt_operation_and_json(self):
        self.assertIn("参数错误", await self.call("save_config", {"output_path": "edit.json"}))
        await self.add()
        self.assertIn("参数错误", await self.call("save_config", {"output_path": "edit.yaml"}))

    async def test_save_ego_trajectory_offsets_json(self):
        await self.call("set_ego_trajectory_offsets", {
            "offsets": [[0, 0.02, 0], [0, 0.03, 0]],
        })
        saved = json.loads(await self.call("save_ego_trajectory_offsets", {
            "output_path": "ego_offsets.json",
        }))
        payload = json.loads(Path(saved["saved_path"]).read_text(encoding="utf-8"))
        self.assertEqual(payload, {
            "coordinate_system": "camera_local_forward_left_up",
            "offsets": [[0, 0.02, 0], [0, 0.03, 0]],
        })

    async def test_removed_instance_cannot_receive_new_trajectory(self):
        await self.call("remove_rigid_instance", {"instance_id": 0})
        before = await self.preview()
        result = await self.call("edit_rigid_trajectory", {
            "instance_id": 0, "waypoints": [[0, -2, 0]],
        })
        self.assertTrue(result.startswith("参数错误:"))
        self.assertEqual(await self.preview(), before)

    async def test_invalid_removal_id_preserves_draft(self):
        before = await self.preview()
        for args in ({}, *({"instance_id": value} for value in (-1, True, 1.5, "0", None))):
            with self.subTest(args=args):
                result = await self.call("remove_rigid_instance", args)
                self.assertTrue(result.startswith("参数错误:"))
                self.assertEqual(await self.preview(), before)

    async def test_revoke_dggt_insertion(self):
        await self.add()
        result = json.loads(await self.call("remove_added_rigid_asset", {"index": 0}))
        self.assertEqual(result["removed_index"], 0)
        self.assertEqual((await self.preview())["Nodes"], {})
        self.assertTrue(self.car.is_file())

    async def test_invalid_asset_index_preserves_draft(self):
        await self.add()
        before = await self.preview()
        for args in ({}, *({"index": value} for value in (-1, 1, True, 1.5, "0", None))):
            with self.subTest(args=args):
                result = await self.call("remove_added_rigid_asset", args)
                self.assertTrue(result.startswith("参数错误:"))
                self.assertEqual(await self.preview(), before)

    async def test_revoking_last_asset_cleans_empty_sections(self):
        await self.add()
        await self.call("remove_added_rigid_asset", {"index": 0})
        self.assertEqual((await self.preview())["Nodes"], {})
        result = await self.call("remove_added_rigid_asset", {"index": 0})
        self.assertTrue(result.startswith("参数错误:"))
        await self.call("remove_rigid_instance", {"instance_id": 5})
        await self.add()
        await self.call("remove_added_rigid_asset", {"index": 0})
        self.assertEqual((await self.preview())["Nodes"], {
            "RigidNodes": {"remove": {"instance_id": [5]}},
        })


if __name__ == "__main__":
    unittest.main()
