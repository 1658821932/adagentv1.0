"""Load explicit JSON tool-result references, without importing orchestration."""
import json
from pathlib import Path


def load_tool_result(value, expected_tool):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Tool result reference must be an existing absolute JSON file")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read JSON tool result reference") from exc
    if (not isinstance(envelope, dict)
            or envelope.get("tool") != expected_tool
            or envelope.get("status") != "success"
            or not isinstance(envelope.get("result"), dict)):
        raise ValueError("Reference must contain a successful " + expected_tool + " envelope")
    return envelope["result"]
