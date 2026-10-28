"""Natural-language MCP client for the external Smart Edit server."""
from __future__ import annotations
import asyncio, json, os, sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from .config import llm_settings

def _load_prompt(name: str) -> str:
    return (Path(__file__).resolve().parent / "prompts" / (name + ".md")).read_text(encoding="utf-8").strip()

def text_result(result: Any) -> str:
    return "\n".join(item.text for item in result.content if getattr(item, "text", None) is not None)

def _chat_kwargs(messages, openai_tools):
    """Build chat completion kwargs, omitting temperature when unset or unsupported."""
    settings = llm_settings()
    kwargs = {
        "model": settings["model"],
        "messages": messages,
        "tools": openai_tools,
        "tool_choice": "auto",
        "max_tokens": settings["max_tokens"],
    }
    temp = os.environ.get("OPENAI_TEMPERATURE")
    if temp:
        kwargs["temperature"] = float(temp)
    return kwargs

async def run(payload: dict[str, Any]) -> dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from openai import AsyncOpenAI
    # Default to the vendored smart_edit core; SMART_EDIT_ROOT overrides.
    _default = Path(__file__).resolve().parents[1] / "core"
    root = Path(os.environ.get("SMART_EDIT_ROOT", str(_default))).resolve()
    load_dotenv(root / ".env.local", override=False)
    output_path = str(Path(payload["output_path"]).resolve())
    env = os.environ.copy()
    env.update({"ASSETS_DIR": str(root / "assets"), "SMPL_ASSETS_DIR": str(root / "assets" / "pedestrian"), "OUTPUT_DIR": str(Path(output_path).parent)})
    server = StdioServerParameters(command=sys.executable, args=["-m", "smart_edit"], cwd=str(root), env=env)
    settings = llm_settings()
    missing = [key for key in ("api_key", "base_url", "model") if not settings[key]]
    if missing:
        raise ValueError("Missing SmartEdit LLM settings: " + ", ".join(missing))
    client = AsyncOpenAI(api_key=settings["api_key"], base_url=settings["base_url"], timeout=settings["timeout"])
    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        tools = (await session.list_tools()).tools
        openai_tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema or {"type": "object"}}} for t in tools]
        required_asset = payload.get("asset_path")
        asset_instruction = f" The user-selected asset is exactly {required_asset}; use this path for the replacement and do not substitute another asset." if required_asset else ""
        output_kind = payload.get("output_kind", "rigid_config")
        if output_kind == "ego_trajectory_offsets":
            task_instruction = _load_prompt("ego_trajectory_offsets")
            save_tool = "save_ego_trajectory_offsets"
        else:
            task_instruction = _load_prompt("rigid_config")
            save_tool = "save_config"
        messages = [{"role": "system", "content": "You operate the DGGT Smart Edit MCP server. " + task_instruction + " Do not claim success without the required save tool." + asset_instruction}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        calls_seen = []
        saved_path = None
        for _ in range(12):
            response = await client.chat.completions.create(**_chat_kwargs(messages, openai_tools))
            message = response.choices[0].message
            calls = message.tool_calls or []
            if not calls:
                if saved_path: break
                raise RuntimeError(message.content or "Smart Edit finished without save_config")
            messages.append(message.model_dump(exclude_none=True))
            for call in calls:
                try: arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc: raise RuntimeError(f"invalid Smart Edit tool arguments: {exc}") from exc
                result = await session.call_tool(call.function.name, arguments=arguments)
                result_text = text_result(result)
                calls_seen.append(call.function.name)
                if call.function.name == save_tool:
                    try:
                        saved = json.loads(result_text)
                        saved_path = saved.get("saved_path") or (saved.get("saved_paths") or [None])[0]
                    except json.JSONDecodeError: pass
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})
        if not saved_path: raise RuntimeError("Smart Edit did not call save_config")
        return {"saved_path": str(Path(saved_path).resolve()), "tool_calls": calls_seen}

def main() -> None:
    try: print(json.dumps({"status": "success", "result": asyncio.run(run(json.load(sys.stdin)))}, ensure_ascii=False))
    except Exception as exc:
        subs = getattr(exc, "exceptions", None)
        if subs:
            import traceback, sys as _sys
            for sub in subs:
                traceback.print_exception(type(sub), sub, sub.__traceback__, file=_sys.stderr)
            exc = subs[0]
        print(json.dumps({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False)); raise SystemExit(1)

if __name__ == "__main__": main()
