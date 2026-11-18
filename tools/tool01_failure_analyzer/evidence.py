"""Per-call evidence directory for tool 01.

One directory per invocation, keyed by the simulation identity, the scene token and a
unique run id, so repeat runs, other simulations and concurrent calls never share files.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4

LOG_ROOT = Path(__file__).resolve().parents[2] / "logs" / "tool01"
MANIFEST = "manifest.json"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_name(value, limit=60):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).replace("..", "_").strip("._-")
    return (text[:limit] or "unknown")


def identity(value):
    """Readable, filesystem-safe label that still separates equal-looking inputs."""
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]
    return f"{safe_name(value)}-{digest}"


def new_run_dir(simulation, token, *, root=None):
    run_dir = (Path(root or LOG_ROOT) / identity(simulation) / safe_name(token)
               / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid4().hex[:8]}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def manifest_path(run_dir):
    return Path(run_dir) / MANIFEST


def read_manifest(run_dir):
    path = manifest_path(run_dir)
    return json.loads(path.read_text()) if path.is_file() else {}


def write_manifest(run_dir, **updates):
    path = manifest_path(run_dir)
    data = read_manifest(run_dir)
    data.update({key: value for key, value in updates.items() if value is not None})
    data["updated_at"] = stamp()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(temporary, path)
    return data


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sent_image_sha256(url):
    """Hash the bytes actually sent, not the file or sheet they were encoded from."""
    head, _, payload = url.partition(",")
    if not head.startswith("data:image/") or not payload:
        raise ValueError("image url is not an inline data URL")
    return sha256_bytes(base64.b64decode(payload))