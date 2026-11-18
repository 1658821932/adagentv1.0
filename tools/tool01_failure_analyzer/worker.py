"""Dedicated worker for tool 01.

The whole analysis (data subprocess, image preparation, model request and its retries)
runs inside this process, which the parent starts in its own session so that a single
group kill stops everything and leaves no request or image writing behind.
"""
from __future__ import annotations

import json
import sys

from tools.tool01_failure_analyzer import real


def main():
    request = json.load(sys.stdin)
    result_stream = sys.stdout
    sys.stdout = sys.stderr  # library chatter must never reach the result stream
    try:
        result = real.analyze(request["input_data"], request["run_dir"])
    except BaseException as error:  # the parent still needs one parseable object
        result = dict(status="error", tool="failure_analyzer",
                      error=dict(type=type(error).__name__, stage="worker", message=str(error)))
    finally:
        sys.stdout = result_stream
    json.dump(result, result_stream, ensure_ascii=False)
    result_stream.write("\n")
    result_stream.flush()


if __name__ == "__main__":
    main()