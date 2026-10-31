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


def type_of(prop: dict, refs: list | None = None) -> str:
    if "$ref" in prop:
        name = prop["$ref"].rsplit("/", 1)[-1]
        if refs is not None and name not in refs:
            refs.append(name)
        return name
    if "enum" in prop:
        return "enum[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in prop["enum"]) + "]"
    if "anyOf" in prop:
        parts = []
        for p in prop["anyOf"]:
            t = type_of(p, refs)
            if t not in parts:
                parts.append(t)
        return " \\| ".join(parts)
    t = prop.get("type")
    if t == "array":
        items = prop.get("items")
        return f"array<{type_of(items, refs)}>" if items else "array"
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
    info = model.model_fields.get(field) if model is not None else None
    alias = getattr(info, "validation_alias", None) if info else None
    choices = getattr(alias, "choices", None)
    if not choices:
        return ""
    extras = [c for c in choices if c != field]
    return ("（别名：`" + "`, `".join(extras) + "`）") if extras else ""


def render_table(schema: dict, notes: dict, model) -> tuple[str, list]:
    """Render a field table; returns (markdown, referenced sub-model names)."""
    props = schema.get("properties", {})
    if not props:
        return "（无字段）\n", []
    required = set(schema.get("required", []))
    refs: list = []
    lines = ["| 字段 | 类型 | 必填 | 默认值 | 说明 |", "|---|---|---|---|---|"]
    for name, prop in props.items():
        note = notes.get(name) or prop.get("description") or ""
        note = str(note).replace("|", "\\|").replace("\n", " ") + aliases_of(model, name)
        lines.append(
            f"| `{name}` | {type_of(prop, refs)} | {'✅' if name in required else '—'} "
            f"| {default_of(prop)} | {note} |"
        )
    return "\n".join(lines) + "\n", refs


def render_nested(schema: dict, refs: list, schemas_module) -> list:
    """Expand one level of referenced sub-models below the main table."""
    defs = schema.get("$defs", {})
    out = []
    for name in refs:
        sub = defs.get(name)
        if not sub:
            continue
        sub_model = getattr(schemas_module, name, None)
        md, _ = render_table(sub, {}, sub_model)
        out.append(f"**`{name}` 对象字段：**\n")
        out.append(md)
    return out


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
    count = 0
    for dirname in tool_dirs():
        count += 1
        module = importlib.import_module(f"tools.{dirname}.description")
        schemas = importlib.import_module(f"tools.{dirname}.schemas")
        desc = module.DESCRIPTION
        sections.append(f"## {dirname}: `{desc['name']}`")
        sections.append("")
        sections.append(desc.get("description", "").strip())
        when = desc.get("when_to_use", "").strip()
        if when:
            sections.append("")
            sections.append(f"**何时使用**：{when}")
        sections.append("")
        for label, key, notes_key, model in (
                ("输入", "input_schema", "inputs", schemas.Input),
                ("输出", "output_schema", "outputs", schemas.Output)):
            schema = desc.get(key, {})
            md, refs = render_table(schema, desc.get(notes_key, {}), model)
            sections.append(f"### {label}")
            sections.append("")
            sections.append(md)
            sections.extend(render_nested(schema, refs, schemas))
        sections.append("")
    OUTPUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({count} tools)")


if __name__ == "__main__":
    main()
