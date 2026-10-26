"""显式运行的刚体 API 多轮验收;会调用 .env.local 中配置的模型。"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from clients.terminal_chat import SmartEditTerminalClient, parse_args
from smart_edit.nodes.rigid import RigidNodeBuilder


class AuditedClient(SmartEditTerminalClient):
    def __init__(self, args):
        super().__init__(args)
        self.audit = []
        self.openai = self.openai.with_options(timeout=45.0, max_retries=0)

    async def _call_mcp_tool(self, name, arguments_json):
        arguments, _ = self._parse_tool_arguments(arguments_json)
        result = await super()._call_mcp_tool(name, arguments_json)
        self.audit.append({"tool": name, "arguments": arguments, "result": result})
        return result


def scenarios():
    taxi = str(PROJECT_ROOT / "assets/car/taxi.ply")
    edit9 = {"instance_id": [9], "waypoints": [[[0, 1, 0]]], "method": ["linear"]}
    return {
        "instances": [
            ("新建配置,将已有车辆2向右偏移2米,车辆9向左偏移1米。保存为edit.yaml。",
             {"trajectory_edit": {"instance_id": [2, 9], "waypoints": [[[0, -2, 0]], [[0, 1, 0]]], "method": ["linear", "linear"]}}, "edit.yaml"),
            ("继续这份配置,删除原场景车辆2和3,保留车辆9的轨迹编辑。保存为remove.yaml。",
             {"trajectory_edit": edit9, "remove": {"instance_id": [2, 3]}}, "remove.yaml"),
            ("继续删除车辆2,再删除车辆4。之前的删除和车辆9的轨迹编辑都要保留,保存为repeat.yaml。",
             {"trajectory_edit": edit9, "remove": {"instance_id": [2, 3, 4]}}, "repeat.yaml"),
        ],
        "insertion": [
            ("新建配置,使用taxi.ply在DGGT第一帧像素[259,250]插入车辆,缩放倍数0.02,保存为insert.json。",
             {"add": {"add_obj": [taxi], "anchor_pixel": [[259, 250]], "scale": [0.02]}}, "insert.json"),
            ("把插入位置改为第一帧像素[300,240],缩放倍数改为0.03,保存为replace_insert.json。",
             {"add": {"add_obj": [taxi], "anchor_pixel": [[300, 240]], "scale": [0.03]}}, "replace_insert.json"),
            ("撤销当前插入车辆,保存为空配置empty.json。",
             {}, "empty.json"),
        ],
        "missing_id": [
            ("删除原场景里前方挡路的那辆车。", {}, None),
            ("那辆车的instance_id是7,请生成删除配置并保存为resolved.yaml。",
             {"remove": {"instance_id": [7]}}, "resolved.yaml"),
        ],
    }


def canonical(node):
    node = json.loads(json.dumps(node))
    if "remove" in node:
        node["remove"]["instance_id"].sort()
    if "trajectory_edit" in node:
        edit = node["trajectory_edit"]
        order = sorted(range(len(edit["instance_id"])), key=edit["instance_id"].__getitem__)
        node["trajectory_edit"] = {key: [values[i] for i in order] for key, values in edit.items()}
    return node


def verify(config, expected):
    assert set(config["Nodes"]) <= {"RigidNodes"}, config
    node = config["Nodes"].get("RigidNodes", {})
    RigidNodeBuilder().validate(node)
    assert canonical(node) == canonical(expected), {"expected": expected, "actual": node}


async def run_case(args, report, run_dir, name, turns, round_index):
    args.output_dir = str(run_dir / f"round-{round_index}" / name)
    args.show_tool_process = False
    client = AuditedClient(args)
    async with client:
        for turn_index, (prompt, expected, filename) in enumerate(turns, 1):
            start = len(client.audit)
            result = {"round": round_index, "scenario": name, "turn": turn_index, "prompt": prompt}
            try:
                reply = await client.answer(prompt)
                result["reply"] = reply
                result["tools"] = client.audit[start:]
                assert not reply.startswith((
                    "模型请求失败:", "工具调用轮次过多", "检测到模型重复提交",
                )), reply
                snapshot = await client.session.call_tool("preview_config", {})
                config = json.loads(snapshot.content[0].text)
                result["config"] = config
                verify(config, expected)
                if filename:
                    path = Path(args.output_dir) / filename
                    assert path.is_file(), "Expected output was not saved"
                    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
                    verify(saved, expected)
                    assert saved == config, "Saved config differs from active draft"
                    result["saved_path"] = str(path)
                    assert any(call["tool"] == "save_config" for call in result["tools"])
                elif name == "missing_id":
                    assert not any(call["tool"].startswith("remove_") for call in result["tools"])
                    assert "id" in reply.lower() or "编号" in reply, reply
                result["status"] = "PASS"
            except Exception as exc:
                result["status"] = "FAIL"
                result["error"] = str(exc).replace(args.api_key, "[REDACTED]")
                result.setdefault("tools", client.audit[start:])
            report["results"].append(result)
            (run_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f'{result["status"]}: round={round_index} scenario={name} turn={turn_index}', flush=True)
            if result["status"] == "FAIL":
                print(result["error"], flush=True)
                break  # 后续用例依赖本轮状态,失败后单独重试该场景。


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scenario", choices=list(scenarios()))
    options, client_options = parser.parse_known_args()
    if options.rounds < 1:
        parser.error("--rounds must be positive")
    sys.argv = [sys.argv[0], *client_options]
    args = parse_args()
    run_dir = PROJECT_ROOT / "outputs" / (
        "rigid-multiround-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    run_dir.mkdir(parents=True)
    report = {"model": args.model, "base_url": args.base_url, "rounds": options.rounds, "results": []}
    print("REPORT:", run_dir / "report.json", flush=True)
    for round_index in range(1, options.rounds + 1):
        for name, turns in scenarios().items():
            if options.scenario and name != options.scenario:
                continue
            await run_case(args, report, run_dir, name, turns, round_index)
    passed = sum(result["status"] == "PASS" for result in report["results"])
    failed = len(report["results"]) - passed
    print(f"TOTAL: {passed} passed, {failed} failed", flush=True)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    asyncio.run(main())
