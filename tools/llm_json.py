"""Small adapter for tools that need structured output from the project LLM."""
from __future__ import annotations

import json
from typing import Any

def ask_json(input_data: dict[str, Any], system_prompt: str, payload: Any) -> dict[str, Any]:
    client = input_data.get("_llm_client")
    if client is None:
        from agent.llm_client import LLMClient
        client = LLMClient()
    response = client.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ])
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM must return a JSON object") from exc
    if not isinstance(result, dict):
        raise ValueError("LLM JSON response must be an object")
    return result
