"""Exercise the three typed text-LLM tools via the public registry.

Use --mode mock for an offline contract test, or --mode real to call the
configured external text model. Inputs are synthetic; no GPU backend runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from tools.registry import call_tool

ANALYSIS={
    "failure_type":["late_response"],
    "failure_reason":"Synthetic test: delayed response to a slowing lead vehicle.",
    "key_scene_factors":["same-lane lead vehicle","straight road"],
    "risk_factors":["short headway"],
    "supporting_case_ids":[],
}
EDIT_INPUT={
    "edit_request":"Plan a trajectory edit for existing vehicle 0 to move toward the ego lane, then maintain its lane position. Use the task's 40 frames; do not invent calibrated distances or claim actual ego braking.",
    "scene_info":{"num_frames":40,"coordinate_system":"smart_edit","note":"Synthetic contract-test context; no real reconstruction or calibrated trajectory is supplied."},
    "instances":[{"instance_id":0,"category":"car","sample_frames":[{"frame":0},{"frame":20},{"frame":39}]}],
}


def run_smoke(mode: str) -> dict:
    checks=[]
    def check(label,tool,arguments):
        result=call_tool(tool,arguments,mode=mode)
        checks.append({"check":label,"tool":tool,"output":result})
        print(label,result["status"],flush=True)
        return result

    requirement=check("requirements","scene_requirement_generator",{"failure_analysis":ANALYSIS})
    if requirement["status"]=="success":
        check("conception","scene_conceiver",{"scene_requirement":requirement})
    else:
        checks.append({"check":"conception","tool":"scene_conceiver","output":{"status":"skipped","reason":"requirements failed"}})
    check("edit_plan","edit_plan_builder",EDIT_INPUT)
    passed=all(item["output"].get("status")=="success" for item in checks)
    return {
        "mode":mode,"synthetic_inputs":True,"passed":passed,
        "timestamp":datetime.now(timezone.utc).isoformat(),"checks":checks,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["mock","real"],default="mock")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    output=args.output or PROJECT_ROOT/"logs"/"llm_tools_smoke"/(
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+"_"+args.mode+".json")
    report=run_smoke(args.mode)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Report:",output.resolve(),flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":
    main()
