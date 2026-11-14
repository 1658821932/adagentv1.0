"""Native policy backend entry for an editor-owned bounded self-test."""
import contextlib
import json
import sys


def main():
    from .client import run_policy
    request=json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result=run_policy(request)
    print(json.dumps(dict(status='success',result=result),allow_nan=False))


if __name__=='__main__':main()
