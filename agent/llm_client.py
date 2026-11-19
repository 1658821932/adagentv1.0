"""
LLM client for AdAgent.

This module only handles communication with an OpenAI-compatible LLM API.
Agent orchestration is handled by agent/orchestrator.py.
"""

import os
from typing import Dict, List

from api_config import get_profile
from common.llm_audit import RequestAudit
from agent.connection_retry import call_with_connection_retries



class LLMClient:
    """Simple wrapper around an OpenAI-compatible chat API."""

    @staticmethod
    def _load_llm_config(profile_name: str = "kimi") -> Dict[str, object]:
        """Read the project-wide LLM profile selected for this caller."""
        return get_profile(profile_name)

    def __init__(self, component: str | None = None):
        # A tool may pin its own profile and thinking level; anything unpinned
        # follows active_profile / the global reasoning_effort.
        profile_name, effort = "kimi", None
        from common.api_profiles import ENV, effort_for, profile_argument, reasoning_effort_level
        if component:
            if component in ENV:
                profile_name, effort = profile_argument(component), effort_for(component)
        file_config = self._load_llm_config(profile_name)
        self.request_profile = dict(file_config)
        effort = effort or reasoning_effort_level()
        if effort:
            self.request_profile["reasoning_level_override"] = effort
        self.component = component
        self.last_usage = None
        # Optional live sink for streamed deltas: fn(kind, text) with kind in {"reasoning","content"}.
        self.on_delta = None

        self.api_key = os.getenv(file_config.get("api_key_env", "OPENAI_API_KEY"))
        self.base_url = file_config.get("base_url")
        self.model = file_config["model"]
        self.timeout = int(file_config["timeout"])
        self.max_tokens = int(file_config["max_tokens"])

        missing = []
        if not self.api_key:
            missing.append(file_config["api_key_env"])
        if not self.base_url:
            missing.append("configs/api.json: base_url")
        if not self.model:
            missing.append("configs/api.json: model")

        if missing:
            raise EnvironmentError(
                "Missing environment variables: " + ", ".join(missing)
            )

        try:
            from openai import OpenAI, Timeout
        except ImportError as exc:
            raise ImportError(
                "openai package is required to use LLMClient"
            ) from exc

        self.no_output_timeout = float(file_config.get("no_output_timeout", min(60.0, float(self.timeout))))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=Timeout(float(self.timeout), connect=15.0, read=self.no_output_timeout),
            max_retries=0,  # Retry policy belongs to explicit request/task handling.
        )

    def _request_parameters(self, parameters):
        from mcp_servers.scene_editing.core.adversarial.llm_request_policy import apply_request_policy
        return apply_request_policy(getattr(self, 'request_profile', {}), parameters)

    def _notify_connection_retry(self, attempt, delay, error):
        callback = getattr(self, 'on_retry', None)
        if callback is not None:
            callback(attempt, delay, error)
        print(f"LLM {type(error).__name__}; retry {attempt}/3 in {delay:g}s", flush=True)

    def chat(self, messages: List[Dict[str, object]]) -> str:
        return call_with_connection_retries(lambda: self._chat_once(messages),
            notify=self._notify_connection_retry)

    def _chat_once(self, messages: List[Dict[str, object]]) -> str:
        """Send chat messages and return model response text."""
        parameters = self._request_parameters(dict(
            model=self.model, messages=messages, max_tokens=self.max_tokens,
            extra_body={"thinking": {"type": "disabled"}}))
        audit = RequestAudit(parameters, endpoint=getattr(self, 'base_url', None),
                             secrets=[getattr(self, 'api_key', '')])
        self.last_usage = None
        response = None
        try:
            call_kwargs = dict(parameters)
            if getattr(self, "timeout", None) is not None:
                call_kwargs["timeout"] = float(self.timeout)
            response = self.client.chat.completions.create(**call_kwargs)
            audit.first_response()

            usage = getattr(response, "usage", None)
            self.last_usage = usage.model_dump() if hasattr(usage, "model_dump") else None
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned empty response")

        except BaseException as exc:
            audit.finish(response=response, usage=self.last_usage, error=exc)
            raise
        audit.finish(response=response, usage=self.last_usage)
        return content

    def call_functions(self, messages, tools):
        return call_with_connection_retries(lambda: self._call_functions_once(messages, tools),
            metrics=lambda: getattr(self, 'last_request_metrics', {}),
            notify=self._notify_connection_retry)

    def _call_functions_once(self, messages, tools):
        """Collect the entire native stream before any caller may execute a tool."""
        from time import perf_counter
        self.last_usage = None
        self.last_request_metrics = {"stream":True,"chunks":0,"has_output":False}
        calls, content, reasoning = {}, [], []
        finish = None
        has_output = False
        no_output_limit = float(getattr(self, "no_output_timeout", min(60.0, float(getattr(self, "timeout", 300)))))
        parameters = self._request_parameters(dict(
            model=self.model, messages=messages, tools=tools,
            tool_choice="required", parallel_tool_calls=False,
            stream=True, stream_options={"include_usage":True},
            max_tokens=self.max_tokens, extra_body={"thinking":{"type":"disabled"}}))
        self.last_request_metrics['request_parameters'] = {key:parameters.get(key)
            for key in ('model','max_tokens','reasoning_effort','extra_body','tool_choice','parallel_tool_calls')}
        audit = RequestAudit(parameters, endpoint=getattr(self, 'base_url', None),
                             label='native_functions', secrets=[getattr(self, 'api_key', '')])
        started = audit.started
        stream = None
        try:
            stream = self.client.chat.completions.create(**parameters)
            for chunk in stream:
                audit.first_response()
                self.last_request_metrics["chunks"] += 1
                self.last_request_metrics.setdefault("first_chunk_seconds",round(perf_counter()-started,4))
                if getattr(chunk,"usage",None) is not None:
                    self.last_usage = chunk.usage.model_dump()
                for choice in chunk.choices:
                    if choice.index != 0:
                        raise ValueError("Only one completion choice is supported")
                    delta = choice.delta
                    if getattr(delta, 'reasoning_content', None):
                        has_output = True
                        reasoning.append(delta.reasoning_content)
                        if self.on_delta:
                            self.on_delta("reasoning", delta.reasoning_content)
                    if getattr(delta,"content",None):
                        has_output = True
                        content.append(delta.content)
                        if self.on_delta:
                            self.on_delta("content", delta.content)
                    for part in getattr(delta,"tool_calls",None) or []:
                        has_output = True
                        item = calls.setdefault(part.index, {"id":"","type":"function","function":{"name":"","arguments":""}})
                        if part.id:
                            if item["id"] and item["id"] != part.id:
                                raise ValueError("Tool-call ID changed during stream")
                            item["id"] = part.id
                        if part.function:
                            item["function"]["name"] += part.function.name or ""
                            item["function"]["arguments"] += part.function.arguments or ""
                    if choice.finish_reason:
                        finish = choice.finish_reason
                self.last_request_metrics["has_output"] = has_output
                if not has_output and finish is None and (perf_counter() - started) > no_output_limit:
                    raise TimeoutError("Model stream stalled with zero output")
            if finish is None:
                raise RuntimeError("Function-call stream ended without a completion marker; no tool may execute")
            if any(not item["id"] for item in calls.values()):
                raise RuntimeError("Function-call stream has a missing tool-call ID")
            self.last_request_metrics["tool_call_count"] = len(calls)
            message = {"role":"assistant","content":"".join(content) or None,
                       "tool_calls":[calls[i] for i in sorted(calls)]}
            if reasoning or getattr(self, 'request_profile', {}).get('thinking_mode') == 'enabled':
                # Opaque provider continuation data; never added to task/state text.
                message['reasoning_content'] = ''.join(reasoning)
        except BaseException as exc:
            self.last_request_metrics["elapsed_seconds"] = round(perf_counter()-started,4)
            audit.finish(response={"content": ''.join(content), "tool_calls": list(calls.values())},
                         usage=self.last_usage, error=exc, metrics=self.last_request_metrics)
            raise
        else:
            self.last_request_metrics["elapsed_seconds"] = round(perf_counter()-started,4)
            audit.finish(response={"message": message, "finish_reason": finish},
                         usage=self.last_usage, metrics=self.last_request_metrics)
        finally:
            if stream is not None:
                stream.close()
        return {"message":message, "finish_reason":finish}


if __name__ == "__main__":
    client = LLMClient()
    print(
        client.chat([
            {"role": "user", "content": "Hello"}
        ])
    )
