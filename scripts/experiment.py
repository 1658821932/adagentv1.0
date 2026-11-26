#!/usr/bin/env python3
"""Manage per-iteration / per-ablation workspaces under outputs/experiments/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.experiments import (
    init_experiment,
    list_experiments,
    resolve_experiment_paths,
    set_active_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage outputs/experiments/ iterations and ablations.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all experiments and active marker.")

    p_show = sub.add_parser("show", help="Show resolved paths for an experiment (defaults to active).")
    p_show.add_argument("exp_id", nargs="?", default=None)

    p_init = sub.add_parser("init", help="Initialize a new experiment directory.")
    p_init.add_argument("exp_id")
    p_init.add_argument("--parent", default=None)
    p_init.add_argument("--kind", choices=["iteration", "ablation"], default="iteration")
    p_init.add_argument("--notes", default="")
    p_init.add_argument("--activate", action="store_true")

    p_use = sub.add_parser("use", help="Point outputs/experiments/active to exp_id.")
    p_use.add_argument("exp_id")

    args = parser.parse_args()
    if args.cmd == "list":
        rows = list_experiments()
        if not rows:
            print("No experiments under outputs/experiments/.")
            return 0
        for r in rows:
            mark = "*" if r["active"] else " "
            m = r.get("metrics") or {}
            score = f"PDMS={m['PDMS']:.5f} SR={m['SR']:.2f}%" if "PDMS" in m and "SR" in m else "-"
            print(f"{mark} {r['exp_id']:28s} [{r['kind']:9s}] parent={str(r['parent']):22s} {score}")
        return 0
    if args.cmd == "show":
        print(json.dumps(resolve_experiment_paths(args.exp_id), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "init":
        p = init_experiment(args.exp_id, parent=args.parent, kind=args.kind, notes=args.notes)
        if args.activate:
            set_active_experiment(args.exp_id)
        print(f"Initialized {p}")
        return 0
    if args.cmd == "use":
        target = set_active_experiment(args.exp_id)
        print(f"Active experiment -> {target}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
