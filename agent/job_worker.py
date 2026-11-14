"""Detached worker for the common executor; business tools remain unchanged."""
import os
import json
import sys
import time
from pathlib import Path
from agent.jobs import locked, write

def execute_request(request):
    try:
        from tools.registry import call_tool
        from agent.schemas import validate_tool_result
        result = call_tool(request["tool"], request["arguments"], mode=request["mode"])
        valid, errors = validate_tool_result(result, request["tool"])
        if not valid:
            return dict(status="error", tool=request["tool"],
                        error=dict(type="invalid_tool_result", message="; ".join(errors),
                                   validation_errors=errors))
        return result
    except BaseException as exc:
        return dict(status="error", tool=request["tool"],
                    error=dict(type="execution_error", message=str(exc)))

def main():
    folder = Path(sys.argv[1])
    # Lease is held for the entire execution, including result persistence.
    with locked(folder / "lease.lock"):
        with locked(folder / "state.lock"):
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["status"] = "running"
            write(folder / "manifest.json", manifest)
        request = json.loads((folder / "request.json").read_text())
        os.environ["ADAGENT_JOB_DIR"] = str(folder)
        result = execute_request(request)
        with locked(folder / "state.lock"):
            manifest.update(status="completed" if result["status"] == "success" else "failed",
                            finished_at=time.time())
            write(folder / "result.json", result)
            write(folder / "manifest.json", manifest)

if __name__ == "__main__":
    main()
