"""Stage 1: requirement -> structured search plan."""

from __future__ import annotations

import json
import os
from typing import Any

from .common import chat_completion, extract_json, model_settings, load_prompt





SYSTEM_PROMPT = load_prompt("search_plan.md")


def generate_search_plan(
    requirement: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    model, base_url, api_key = model_settings(model, base_url)

    body = chat_completion(
        base_url=base_url,
        api_key=api_key,
        payload={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": requirement},
            ],
        },
    )

    plan = extract_json(body["choices"][0]["message"]["content"])
    plan.setdefault("original_input", requirement)
    plan.setdefault("youtube_queries", [])
    plan.setdefault("bilibili_queries", [])
    plan.setdefault("negative_keywords", [])
    return plan
