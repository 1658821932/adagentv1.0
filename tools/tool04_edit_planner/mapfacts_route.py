"""Evaluate route junctions in order and return the first target-compatible one."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("mapfacts_sim.py")


def run(request):
    completed = subprocess.run([sys.executable, str(SCRIPT)], input=json.dumps(request, ensure_ascii=False),
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else completed.stderr)
    return json.loads(completed.stdout)


def main():
    request = json.load(sys.stdin)
    listing = run({**request, "list_junctions": True})
    candidates = listing.get("route_junctions") or []
    if not candidates:
        print(json.dumps(run(request), ensure_ascii=False))
        return
    attempts = []
    best = None
    for candidate in candidates[:6]:
        fid = candidate["intersection_fid"]
        try:
            payload = run({**request, "force_junction_fid": fid})
        except Exception as error:
            attempts.append({"intersection_fid": fid,
                             "entry_distance_m": candidate["entry_distance_m"],
                             "error": str(error)})
            continue
        checks = payload.get("checks") or {}
        attempts.append({"intersection_fid": fid,
                         "entry_distance_m": candidate["entry_distance_m"],
                         "checks": checks,
                         "reasons": payload.get("reasons") or []})
        if payload.get("available") and all(checks.get(name) == "pass"
                                            for name in ("road", "lanes", "space")):
            payload.setdefault("facts", {})["junction_attempts"] = attempts
            payload["facts"]["selected_route_junction"] = payload["facts"].get("target_junction_fid")
            print(json.dumps(payload, ensure_ascii=False, default=str))
            return
        score = (int(all(checks.get(name) != "fail" for name in ("road", "lanes", "space"))),
                 sum(checks.get(name) == "pass" for name in ("road", "lanes", "space")))
        if best is None or score > best[0]:
            best = (score, payload)
    selected = best[1] if best is not None else listing
    selected.setdefault("facts", {})["junction_attempts"] = attempts
    print(json.dumps(selected, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
