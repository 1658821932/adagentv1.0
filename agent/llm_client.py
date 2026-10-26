"""
LLM client for adagentv1.0.

This module only handles communication with an OpenAI-compatible LLM API.
Agent orchestration and tool calling logic will be implemented separately.
"""

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class LLMClient:
    """Simple wrapper around an OpenAI-compatible chat API."""

    def __init__(self):
        # Prefer the standard OpenAI-compatible names; retain the older LLM_*
        # names for existing Agent deployments.
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL")
        self.timeout = int(os.getenv("LLM_TIMEOUT") or os.getenv("OPENAI_TIMEOUT", "600"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))

        missing = []
        if not self.api_key:
            missing.append("OPENAI_API_KEY (or LLM_API_KEY)")
        if not self.base_url:
            missing.append("OPENAI_BASE_URL (or LLM_BASE_URL)")
        if not self.model:
            missing.append("OPENAI_MODEL (or LLM_MODEL)")

        if missing:
            raise EnvironmentError(
                "Missing environment variables: " + ", ".join(missing)
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required to use LLMClient"
            ) from exc

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def chat(self, messages: List[Dict[str, object]]) -> str:
        """Send chat messages and return model response text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned empty response")

            return content

        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc


if __name__ == "__main__":
    client = LLMClient()
    print(
        client.chat([
            {"role": "user", "content": "Hello"}
        ])
    )
