"""JSON entry point for the AdAgent failure-analysis adapter."""
import json
import sys
from dataclasses import asdict
from .service import analyze_scene

def main():
    try:
        result = analyze_scene(**json.load(sys.stdin))
        print(json.dumps({"status": "success", "result": asdict(result)}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
