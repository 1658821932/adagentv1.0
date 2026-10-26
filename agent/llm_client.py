"""
LLM client for adagentv1.0.

This module only handles communication with an OpenAI-compatible LLM API.
Agent orchestration and tool calling logic will be implemented separately.
"""

import os
import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agent.json"


class LLMClient:
    """Simple wrapper around an OpenAI-compatible chat API."""

    @staticmethod
    def _load_llm_config() -> Dict[str, object]:
        """Read the non-sensitive llm block from configs/agent.json."""

        try:
            with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}
        llm = data.get("llm", {})
        return llm if isinstance(llm, dict) else {}

    def __init__(self):
        # Non-sensitive connection settings come from configs/agent.json;
        # environment variables still take precedence when set. Only the API
        # key is expected in .env.
        file_config = self._load_llm_config()

        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or file_config.get("base_url")
        )
        self.model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("LLM_MODEL")
            or file_config.get("model")
        )
        self.timeout = int(
            os.getenv("LLM_TIMEOUT")
            or os.getenv("OPENAI_TIMEOUT")
            or file_config.get("timeout", 600)
        )
        self.max_tokens = int(
            os.getenv("OPENAI_MAX_TOKENS") or file_config.get("max_tokens", 1024)
        )

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
