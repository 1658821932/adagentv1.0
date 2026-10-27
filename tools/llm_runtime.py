"""Shared execution contract for small, text-only LLM tools.

No credentials or model clients are loaded at import time. Tool descriptions,
mock implementations and real implementations use the same typed schemas.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

NonEmptyStr = Annotated[str, Field(min_length=1)]


class ToolModel(BaseModel):
    """Reject misspelled fields and implicit type coercion."""
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class LLMToolConfig(ToolModel):
    model_role: NonEmptyStr
    prompt_version: NonEmptyStr
    timeout: Annotated[int, Field(gt=0)] | None = None
    max_tokens: Annotated[int, Field(gt=0)] | None = None
    validation_retries: Annotated[int, Field(ge=0, le=2)] = 1


class LLMRoleConfig(ToolModel):
    model: NonEmptyStr | None = None
    timeout: Annotated[int, Field(gt=0)] | None = None
    max_tokens: Annotated[int, Field(gt=0)] | None = None


def unwrap_result(value: Any, field: str | None = None) -> Any:
    """Accept a previous result or its standard envelope, without mutating it."""
    for _ in range(3):
        if not isinstance(value, dict):
            break
        if "status" in value and value["status"] != "success":
            raise ValueError("Cannot consume a failed upstream tool result")
        if "result" in value:
            value = value["result"]
        elif field and field in value and isinstance(value[field], dict):
            value = value[field]
        else:
            break
    return value


def validation_details(error: Exception) -> list[dict[str, Any]]:
    if isinstance(error, ValidationError):
        return [
            {"loc": list(item["loc"]), "type": item["type"], "message": item["msg"]}
            for item in error.errors(include_input=False, include_url=False)
        ]
    return [{"loc": [], "type": "value_error", "message": str(error)}]


def parse_json_object(response: str) -> dict[str, Any]:
    if not isinstance(response, str):
        raise ValueError("LLM response must be text")
    text = response.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != fence:
            raise ValueError("LLM returned an incomplete fenced JSON response")
        text = "\n".join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")
    return result


def make_description(
    *, name: str, description: str, when_to_use: str,
    input_model: type[BaseModel], output_model: type[BaseModel],
) -> dict[str, Any]:
    """Generate advertised parameter fields from the executable contracts."""
    input_schema = input_model.model_json_schema()
    output_schema = output_model.model_json_schema()
    def fields(schema):
        return {
            key: value.get("description", value.get("title", key))
            for key, value in schema.get("properties", {}).items()
        }
    return {
        "name": name, "description": description, "when_to_use": when_to_use,
        "inputs": fields(input_schema), "outputs": fields(output_schema),
        "input_schema": input_schema, "output_schema": output_schema,
    }


@dataclass(frozen=True)
class LLMTool:
    name: str
    directory: Path
    input_model: type[BaseModel]
    response_model: type[BaseModel]
    output_model: type[BaseModel] | None = None
    payload_builder: Callable[[BaseModel], Any] | None = None
    enrich_result: Callable[[BaseModel, dict], dict] | None = None
    validate_result: Callable[[BaseModel, BaseModel], None] | None = None

    def _validate(self, data: BaseModel, response: dict) -> dict:
        parsed = self.response_model.model_validate(response)
        result = parsed.model_dump(mode="json")
        if self.enrich_result:
            result = self.enrich_result(data, result)
        output = (self.output_model or self.response_model).model_validate(result)
        if self.validate_result:
            self.validate_result(data, output)
        return output.model_dump(mode="json")

    def execute(self, input_data: dict, *, mock_result: Callable | None = None) -> dict:
        """Validate -> call -> validate -> bounded correction -> envelope."""
        started = perf_counter()
        mode = "mock" if mock_result is not None else "real"
        meta = {"mode": mode, "model": "mock" if mode == "mock" else None, "attempts": 0}

        def error(kind, message, details=None):
            message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(message))
            message = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", message)
            payload = {"type": kind, "message": message}
            if details:
                payload["validation_errors"] = details
            return {
                "status": "error", "tool": self.name, "error": payload,
                "meta": {**meta, "elapsed_seconds": round(perf_counter() - started, 4)},
            }

        try:
            if not isinstance(input_data, dict):
                raise ValueError("Tool input must be a JSON object")
            arguments = dict(input_data)
            injected_client = arguments.pop("_llm_client", None)
            data = self.input_model.model_validate(arguments)
        except (ValueError, TypeError) as exc:
            return error("input_validation_error", "Tool input does not match its schema",
                         validation_details(exc))

        try:
            config = LLMToolConfig.model_validate_json(
                (self.directory / "config.json").read_text(encoding="utf-8")
            )
            prompt = (self.directory / "prompt.md").read_text(encoding="utf-8")
            if not prompt.strip():
                raise ValueError("Tool prompt.md must not be empty")
            meta.update(prompt_version=config.prompt_version,
                        prompt_sha256=sha256(prompt.encode()).hexdigest())
        except (OSError, ValueError) as exc:
            return error("configuration_error", exc)

        if mock_result is not None:
            try:
                result = self._validate(data, mock_result(data))
            except (ValueError, TypeError) as exc:
                return error("output_validation_error", "Mock output violates the tool contract",
                             validation_details(exc))
            return {"status": "success", "tool": self.name, "result": result,
                    "meta": {**meta, "elapsed_seconds": round(perf_counter() - started, 4)}}

        try:
            if injected_client is not None:
                client = injected_client
            else:
                from agent.llm_client import LLMClient
                roles = LLMClient._load_llm_config().get("roles", {})
                if not isinstance(roles, dict) or config.model_role not in roles:
                    raise ValueError(f"Unknown LLM model role: {config.model_role}")
                overrides = LLMRoleConfig.model_validate(roles[config.model_role]).model_dump(exclude_none=True)
                for key in ("timeout", "max_tokens"):
                    value = getattr(config, key)
                    if value is not None:
                        overrides[key] = value
                client = LLMClient(overrides=overrides)
            meta["model"] = getattr(client, "model", "injected-client")
            payload = self.payload_builder(data) if self.payload_builder else data.model_dump(mode="json")
            schema = self.response_model.model_json_schema()
            messages = [
                {"role": "system", "content": prompt.strip()
                 + "\n\nReturn ONLY one JSON object matching this JSON Schema:\n"
                 + json.dumps(schema, ensure_ascii=False)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
        except (OSError, ValueError, TypeError, ImportError) as exc:
            return error("configuration_error", exc)

        for attempt in range(config.validation_retries + 1):
            meta["attempts"] = attempt + 1
            try:
                response = client.chat(messages)
            except Exception as exc:
                # Do not turn auth/network errors into JSON-correction requests.
                return error("llm_request_error", exc)
            try:
                result = self._validate(data, parse_json_object(response))
            except (ValueError, TypeError) as exc:
                details = validation_details(exc)
                if attempt == config.validation_retries:
                    return error("output_validation_error",
                                 "LLM output failed schema or business validation", details)
                messages.extend([
                    {"role": "assistant", "content": response if isinstance(response, str) else ""},
                    {"role": "user", "content":
                     "Correct the previous output using only the original input. "
                     "Do not invent evidence to satisfy validation. Return ONLY JSON. Errors:\n"
                     + json.dumps(details, ensure_ascii=False)},
                ])
                continue
            usage = getattr(client, "last_usage", None)
            if isinstance(usage, dict):
                meta["last_call_usage"] = usage
            return {"status": "success", "tool": self.name, "result": result,
                    "meta": {**meta, "elapsed_seconds": round(perf_counter() - started, 4)}}
        raise AssertionError("Unreachable LLM retry state")
