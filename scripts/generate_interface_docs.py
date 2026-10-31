#!/usr/bin/env python3
"""Generate docs/INTERFACES.md from the tool contracts.

Run from the repository root with the project test interpreter:

    environments/video_retriever/.venv/bin/python scripts/generate_interface_docs.py

Regenerate whenever a tools/toolXX_*/schemas.py or description.py changes.
The generated file is committed; do not edit it by hand.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "docs" / "INTERFACES.md"


def tool_dirs():
    for path in sorted((ROOT / "tools").glob("tool*/description.py")):
        yield path.parent.name


def type_of(prop: dict, defs: dict) -> str:
    if "$ref" in prop:
        return prop["$ref"].rsplit("/", 1)[-1]
    if "enum" in prop:
        return "enum[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in prop["enum"]) + "]"
    if "anyOf" in prop:
        parts = []
        for p in prop["anyOf"]:
            t = type_of(p, defs)
            if t not in parts:
                parts.append(t)
        return " \\| ".join(parts)
    t = prop.get("type")
    if t == "array":
        items = prop.get("items")
        return f"array<{type_of(items, defs)}>" if items else "array"
    if t == "object":
        return "object"
    if t in {"integer", "number"}:
        bound = ""
        if "exclusiveMinimum" in prop:
            bound = f" >{prop['exclusiveMinimum']}"
        elif "minimum" in prop:
            bound = f" >={prop['minimum']}"
        return t + bound
    return t or "any"


def default_of(prop: dict) -> str:
    if "default" not in prop:
        return "—"
    value = prop["default"]
    if value is None:
        return "null"
    return f"`{json.dumps(value, ensure_ascii=False)}`"


def aliases_of(model, field: str) -> str:
    info = model.model_fields.get(field)
    alias = getattr(info, "validation_alias", None) if info else None
    choices = getattr(alias, "choices", None)
    if not choices:
        return ""
    extras = [c for c in choices if c != field]
    return ("（别名：`" + "`, `".join(extras) + "`）") if extras else ""


def render_table(schema: dict, notes: dict, model) -> str:
    props = schema.get("properties", {})
    if not props:
        return "（无字段）\n"
    required = set(schema.get("required", []))
    defs = schema.get("$defs", {})
    lines = ["| 字段 | 类型 | 必填 | 默认值 | 说明 |", "|---|---|---|---|---|"]
    for name, prop in props.items():
        note = notes.get(name) or prop.get("description") or ""
        note = str(note).replace("|", "\\|").replace("\n", " ") + aliases_of(model, name)
        lines.append(
            f"| `{name}` | {type_of(prop, defs)} | {'✅' if name in required else '—'} "
            f"| {default_of(prop)} | {note} |"
        )
    return "\n".join(lines) + "\n"


def main():
    sections = [
        "# 工具接口参考",
        "",
        "> 本文件由 `scripts/generate_interface_docs.py` 自动生成，请勿手工编辑。",
        "> 修改工具契约（`tools/toolXX_*/schemas.py` 或 `description.py`）后重新生成：",
        "> ",
        ">     environments/video_retriever/.venv/bin/python scripts/generate_interface_docs.py",
        "",
        "每个工具的用途和调用时机见 [tools 目录与 13 个 Tool 说明](TOOLS.md)；",
        "调用方式、mock/real 边界和错误类型见 [工具接口与运行管理](EXECUTION_CONTRACTS.md)。",
        "本文档只列字段级契约。",
        "",
    ]
    for dirname in tool_dirs():
        module = importlib.import_module(f"tools.{dirname}.description")
        schemas = importlib.import_module(f"tools.{dirname}.schemas")
        desc = module.DESCRIPTION
        title = f"## {dirname}: `{desc['name']}`"
        sections.append(title)
        sections.append("")
        sections.append(desc.get("description", "").strip())
        when = desc.get("when_to_use", "").strip()
        if when:
            sections.append("")
            sections.append(f"**何时使用**：{when}")
        sections.append("")
        sections.append("### 输入")
        sections.append("")
        sections.append(render_table(desc.get("input_schema", {}), desc.get("inputs", {}), schemas.Input))
        sections.append("### 输出")
        sections.append("")
        sections.append(render_table(desc.get("output_schema", {}), desc.get("outputs", {}), schemas.Output))
        sections.append("")
    OUTPUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(tool_dirs().__class__(list(tool_dirs()))) if False else len(list(tool_dirs()))} tools)")


if __name__ == "__main__":
    main()
