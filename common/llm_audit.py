"""Run-scoped API audit. Standard library only; never changes API payloads.

Enabled by trace_context(agent_log_dir=...) or ADAGENT_AGENT_LOG_DIR. Each
physical request owns a directory, so threads/processes never append together.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from common.trace_io import TRACE, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def trace():
    fields = ("agent_log_dir", "agent_run_id", "tool_call_id", "stage", "step", "decision_attempt", "llm_attempt")
    inherited = {key: os.environ["ADAGENT_" + key.upper()] for key in fields
                 if os.getenv("ADAGENT_" + key.upper())}
    return {**inherited, **TRACE.get()}


class Sanitizer:
    def __init__(self, root, secrets=()):
        self.root = Path(root)
        env_secrets = [value for name, value in os.environ.items()
                       if any(part in name.upper() for part in
                              ("API_KEY", "ACCESS_TOKEN", "SECRET", "PASSWORD")) and value]
        self.secrets = sorted(set(str(s) for s in (*env_secrets, *secrets) if s), key=len, reverse=True)

    def text(self, value):
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", value, flags=re.I)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        # Signed media URLs can contain temporary credentials. Keep their path
        # and explicitly mark that the query was removed.
        def url(match):
            raw = match.group(0)
            parts = urlsplit(raw)
            netloc = parts.netloc.rsplit("@", 1)[-1]
            return urlunsplit((parts.scheme, netloc, parts.path,
                               "[REDACTED_QUERY]" if parts.query else "", ""))
        return re.sub(r"https?://[^\s\"<>]+", url, value)

    def clean(self, value):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if isinstance(value, dict):
            out = {}
            for key, child in value.items():
                low = str(key).lower()
                if low in {"reasoning", "reasoning_content", "encrypted_content", "signature"}:
                    # Provider continuation data is opaque, not task context.
                    out[str(key)] = {"omitted": "provider_private_continuation",
                                     "characters": len(str(child or ""))}
                elif any(part in low for part in ("api_key", "apikey", "authorization", "password", "access_token", "secret", "credential")):
                    out[str(key)] = "[REDACTED]"
                else:
                    out[str(key)] = self.clean(child)
            return out
        if isinstance(value, (list, tuple)):
            return [self.clean(child) for child in value]
        if isinstance(value, str):
            if value.startswith("data:") and ";base64," in value:
                prefix, encoded = value.split(",", 1)
                try:
                    binary = base64.b64decode(encoded, validate=True)
                except ValueError:
                    return {"omitted": "invalid_data_url", "characters": len(value)}
                digest = hashlib.sha256(binary).hexdigest()
                asset = self.root / "assets" / (digest + ".bin")
                asset.parent.mkdir(parents=True, exist_ok=True)
                if not asset.exists():
                    temporary = asset.with_name(asset.name + "." + uuid4().hex + ".tmp")
                    temporary.write_bytes(binary)
                    temporary.replace(asset)
                return {"asset": str(asset.relative_to(self.root)), "sha256": digest,
                        "data_url_prefix": prefix, "bytes": len(binary)}
            return self.text(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(type(value).__name__)


def input_metrics(payload):
    messages = payload.get("messages", [])
    images = 0
    text_chars = 0
    def visit(value):
        nonlocal images, text_chars
        if isinstance(value, dict):
            if value.get("type") in {"image_url", "image", "input_image"}:
                images += 1
                return
            for key, child in value.items():
                if key not in {"reasoning_content", "reasoning", "encrypted_content"}:
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            if not value.startswith("data:"):
                text_chars += len(value)
    visit(messages or payload.get("input", {}))
    encoded = json.dumps(payload, ensure_ascii=False, default=str).encode()
    return {"message_count": len(messages), "image_count": images,
            "text_characters": text_chars, "request_bytes": len(encoded),
            "tool_schema_bytes": len(json.dumps(payload.get("tools", []), ensure_ascii=False).encode()),
            "token_count_source": "provider_usage_when_available"}


class RequestAudit:
    def __init__(self, payload, *, endpoint=None, label="chat", secrets=(), archive_ref=None, logical_key=None):
        self.context = trace()
        self.root = Path(self.context["agent_log_dir"]) if self.context.get("agent_log_dir") else None
        self.folder = None
        self.record = {}
        if self.root:
            self.sanitizer = Sanitizer(self.root, secrets)
            request_id = uuid4().hex
            self.folder = self.root / "api_requests" / request_id
            self.folder.mkdir(parents=True, exist_ok=False)
            tool_id = self.context.get("tool_call_id")
            stage = tool_id.rsplit(":", 1)[-1] if tool_id else self.context.get("stage", "main_agent")
            self.record = dict(schema_version="api_audit.v1", request_id=request_id,
                               context=self.context, stage=stage, label=label,
                               model=payload.get("model"), started_at=utc(), status="running",
                               input_metrics=input_metrics(payload), archive_ref=archive_ref)
            explicit_attempt = re.search(r'/attempt_(\d+)$', label)
            self.record['attempt_index'] = int(self.context.get('llm_attempt') or
                self.context.get('decision_attempt') or (explicit_attempt.group(1) if explicit_attempt else 1))
            if logical_key is not None:
                logical_id = hashlib.sha256(str(logical_key).encode()).hexdigest()
                previous = 0
                for path in (self.root/'api_requests').glob('*/result.json'):
                    record = json.loads(path.read_text())
                    previous += record.get('logical_request_id') == logical_id
                self.record.update(logical_request_id=logical_id, attempt_index=previous+1)
            write_json(self.folder / "request.json", self.sanitizer.clean({"endpoint": endpoint, "payload": payload}))
            write_json(self.folder / "result.json", self.sanitizer.clean(self.record))
        # Exclude local request archival from API time, including large images.
        self.started = time.perf_counter()

    def first_response(self):
        self.record.setdefault("first_response_seconds", round(time.perf_counter() - self.started, 6))

    def finish(self, *, response=None, usage=None, error=None, http_status=None, metrics=None):
        elapsed = round(time.perf_counter() - self.started, 6)
        if not self.folder:
            return
        status = "success" if error is None and (http_status is None or http_status < 400) else "error"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
        self.record.update(status=status, finished_at=utc(), elapsed_seconds=elapsed,
                           http_status=http_status or getattr(error, "status_code", None),
                           error_type=type(error).__name__ if error is not None else None,
                           usage=usage, metrics=metrics or {})
        if error is not None:
            body = getattr(error, 'body', None)
            provider = body.get('error', body) if isinstance(body, dict) else {}
            if isinstance(provider, dict):
                self.record['provider_error'] = {key: provider[key] for key in ('code', 'type', 'param')
                    if isinstance(provider.get(key), (str, int, float)) and len(str(provider[key])) <= 200}
        if response is not None:
            write_json(self.folder / "response.json", self.sanitizer.clean(response))
        write_json(self.folder / "result.json", self.sanitizer.clean(self.record))


def audited_json_call(call, payload, *, endpoint=None, label="chat", secrets=(), return_response=False):
    """One nonstreaming network attempt, including reading its JSON response."""
    audit = RequestAudit(payload, endpoint=endpoint, label=label, secrets=secrets)
    body = None
    status = None
    try:
        response = call()
        audit.first_response()
        status = getattr(response, "status_code", getattr(response, "status", None))
        # SDK models also have .json(), but that returns a JSON *string*.
        # Keep their typed response intact for existing callers.
        body = response.json() if not hasattr(response, "model_dump") and hasattr(response, "json") else response
    except BaseException as error:
        audit.finish(response=body, error=error, http_status=status)
        raise
    usage = body.get("usage") if isinstance(body, dict) else getattr(body, "usage", None)
    provider_error = isinstance(body, dict) and (body.get("error") or body.get("code"))
    audit.finish(response=body, usage=usage, http_status=status,
                 error=RuntimeError("Provider error") if provider_error else None)
    return (response, body) if return_response else body


def sum_usage(items):
    """Sum numeric usage details (including cached/reasoning token subfields)."""
    result = {}
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, dict):
                result[key] = sum_usage([result.get(key, {}), value])
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = result.get(key, 0) + value
    return result


def record_tool_exchange(call, result, archive_ref, *, elapsed_seconds=None):
    context = trace()
    if not context.get('agent_log_dir'):
        return
    root = Path(context['agent_log_dir'])
    # Stable identity prevents counting an archived exchange twice on resume.
    identity = hashlib.sha256(str(Path(archive_ref).resolve()).encode()).hexdigest()
    record = dict(context=context, timestamp=utc(), archive_ref=str(archive_ref), elapsed_seconds=elapsed_seconds,
                  tool_name=call.get('function', {}).get('name'), call=call, result=result)
    write_json(root / 'tool_events' / (identity + '.json'), Sanitizer(root).clean(record))


def record_operation(record, archive_ref):
    context = trace()
    if context.get('agent_log_dir'):
        root = Path(context['agent_log_dir'])
        write_json(root / 'operations' / (record['event_id'] + '.json'),
                   Sanitizer(root).clean(dict(record, context=context, archive_ref=str(archive_ref))))
