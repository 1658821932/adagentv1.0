"""Service layer: invoke worldengine_failure_analysis (wfa) scene analysis.

Wraps wfa's run.py CLI for a single scene token and returns the parsed
VLM forensic result. API settings come from root configs/api.json; credentials from root .env.

A scene can be simulated more than once. ``simulation`` selects WHICH closed-loop
simulation record (archive) produces the official metrics; ``token`` selects the scene
inside that record.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from api_config import API_CONFIG_PATH, load_credentials
from mcp_servers.process import run_logged
from mcp_servers.runtime import runtime_task, current_run

from .config import WFA_ROOT, DEFAULT_CONFIG, DEFAULT_CONFIG_EXAMPLE, DEFAULT_RUN_NAME, _resolve_python, runtime_env
from .schemas import WfaSceneResult

from common.pdm_scores import score_file as _official_score_file


class WfaError(RuntimeError):
    """The native failure-analysis workflow failed."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def _closed_loop_root() -> Path:
    """Directory holding closed-loop experiment results (may be overridden per config)."""
    env = os.getenv("WFA_CLOSED_LOOP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    try:
        base = json.loads(DEFAULT_CONFIG.read_text()).get("result_root")
    except Exception:
        base = None
    if base:
        path = Path(base).expanduser().resolve()
        # <experiments>/closed_loop_exps/<experiment>/<batch>
        for parent in (path, *path.parents):
            if parent.name == "closed_loop_exps":
                return parent
        return path.parent
    return Path("/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/WorldEngine/experiments/closed_loop_exps")


def _is_archive(path: Path) -> bool:
    return path.is_dir() and (path / "scene_index.csv").is_file()


def resolve_simulation(spec: str) -> Path:
    """Resolve a WE simulation record spec to its archive directory.

    Accepts an absolute archive dir, ``<experiment>/<batch>``, or a unique batch name.
    """
    spec = str(spec).strip()
    if not spec:
        raise ValueError("simulation must not be empty")
    raw = Path(spec).expanduser()
    if raw.is_absolute():
        if _is_archive(raw):
            return raw.resolve()
        raise WfaError(f"simulation archive not found or incomplete (needs scene_index.csv): {raw}")

    root = _closed_loop_root()
    direct = root / raw
    if _is_archive(direct):
        return direct.resolve()

    # Accept a bare batch name or a partial suffix, but only when it is unambiguous.
    matches = sorted(
        {p.resolve() for p in root.glob("*/*") if p.is_dir() and p.name == raw.name and _is_archive(p)}
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise WfaError(
            "simulation %r matches multiple archives; pass '<experiment>/<batch>': %s"
            % (spec, ", ".join(str(m) for m in matches))
        )
    raise WfaError(
        "simulation archive %r not found under %s (expected <experiment>/<batch> containing scene_index.csv)"
        % (spec, root)
    )


def _archive_tokens(root: Path) -> list[str]:
    with (root / "scene_index.csv").open(newline="") as stream:
        tokens = [row["token"] for row in csv.DictReader(stream) if row.get("token")]
    if not tokens:
        raise WfaError(f"simulation archive has an empty scene inventory: {root}")
    return tokens


def _config_for_simulation(base_config: Path, root: Path, task) -> Path:
    """Materialise a temporary wfa config pointing at one simulation archive."""
    try:
        official = _official_score_file(root)
    except FileNotFoundError as error:
        raise WfaError(
            "simulation archive is missing official metrics "
            f"(all_scenes_pdm_averages_*.csv): {root}"
        ) from error
    config = json.loads(base_config.read_text())
    config["result_root"] = str(root)
    # Authorisation follows the chosen archive instead of a hardcoded token list.
    config["vlm_allowed_tokens"] = _archive_tokens(root)
    # One resolved table feeds both the adapter gate and the analysis core.
    config["official_metrics_csv"] = str(official)
    target = task.input / "simulation_config.json"
    target.write_text(json.dumps(config, ensure_ascii=False, indent=1))
    return target


@runtime_task("worldengine_failure_analysis", "analyze_scene")
def analyze_scene(
    token: str,
    *,
    simulation: str | None = None,
    config_path: str | None = None,
    run_name: str = DEFAULT_RUN_NAME,
    timeout: int = 300,
) -> WfaSceneResult:
    """Run wfa forensic analysis for one scene token and return the parsed result."""
    if not token or not token.strip():
        raise ValueError("failure_analyzer (wfa) requires a non-empty scene 'token'")

    config = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG
    if not config.is_file():
        if config == DEFAULT_CONFIG and DEFAULT_CONFIG_EXAMPLE.is_file():
            raise WfaError(
                f"wfa config not found: {config}. "
                f"Copy {DEFAULT_CONFIG_EXAMPLE} to {config} and fill in your paths."
            )
        raise WfaError(f"wfa config not found: {config}")

    task = current_run("worldengine_failure_analysis")
    archive: Path | None = None
    if simulation:
        archive = resolve_simulation(simulation)
        config = _config_for_simulation(config, archive, task)
        tokens = _archive_tokens(archive)
        if not any(t == token.strip() or t.endswith("-" + token.strip()) for t in tokens):
            raise WfaError(
                f"scene token {token!r} is not part of simulation {archive.name!r} "
                f"({len(tokens)} scenes); pick another simulation record or token"
            )

    run_py = WFA_ROOT / "run.py"
    cmd = [
        _resolve_python(), str(run_py), "scene",
        "--token", token.strip(),
        "--config", str(config),
        "--api-config", str(API_CONFIG_PATH),
        "--run-name", run_name,
        "--run-dir", str(task.output),
    ]

    load_credentials()
    child = runtime_env(task.child_env())
    from common.api_profiles import effort_for, profile_argument
    child['WFA_LLM_PROFILE'] = profile_argument('failure_analyzer')
    level = effort_for('failure_analyzer')
    child.pop('WFA_REASONING_EFFORT', None)
    if level:
        child['WFA_REASONING_EFFORT'] = level
    proc = run_logged(
        cmd, cwd=str(WFA_ROOT), log_path=task.logs / "analysis.log", timeout=timeout, env=child,
    )

    # The analysis result comes back as one log line: RESULT {json}
    log_path = task.logs / "analysis.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    result = None
    for line in log_text.splitlines():
        if line.startswith("RESULT "):
            try:
                result = json.loads(line[len("RESULT "):])
            except json.JSONDecodeError as exc:
                raise WfaError(f"wfa returned malformed RESULT line: {exc}") from exc

    scene_dir = task.output / "scenes" / token.strip()
    status_path = scene_dir / "status.json"
    if not status_path.is_file():
        scenes_root = task.output / "scenes"
        if scenes_root.is_dir():
            matches = [d for d in scenes_root.iterdir()
                       if d.is_dir() and (d.name == token.strip() or d.name.endswith("-" + token.strip()))]
            if len(matches) == 1:
                scene_dir = matches[0]
                status_path = scene_dir / "status.json"

    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}

    if proc.returncode or status.get('state') == 'failed':
        backend_type = str(status.get('error_type') or 'process_exit')
        cause = str(status.get('message') or f'wfa exited with code {proc.returncode}')[:1500]
        environment = backend_type in {'ImportError', 'ModuleNotFoundError'} or any(
            text in cause for text in ('libGL.so', 'cannot open shared object file', 'No module named'))
        details = dict(failure_stage='environment' if environment else 'backend',
                       backend_error_type=backend_type, log_path=str(log_path), run_dir=str(task.root),
                       archive_validated=archive is not None, scene_validated=archive is not None)
        if environment:
            details['retryable'] = False
        raise WfaError(f'wfa scene analysis failed for {token!r}: {backend_type} {cause}', **details)

    if result is None:
        if status.get("state") == "failed":
            raise WfaError(
                f"wfa scene analysis failed for {token!r}: "
                f"{status.get('error_type')} {status.get('message')}"
            )
        tail = log_text[-1500:]
        raise WfaError(f"wfa produced no RESULT for token {token!r} (exit={proc.returncode}). {tail}")

    return WfaSceneResult(
        token=result.get("token", token.strip()),
        run_dir=str(task.root),
        status=result.get("status", status.get("state", "unknown")),
        failure_reason=result.get("failure_reason", ""),
        reason_category=result.get("reason_category", ""),
        observations=result.get("observations", []),
        candidate_assessment=result.get("candidate_assessment", []),
        uncertainty=result.get("uncertainty", ""),
        official_metrics=result.get("official_metrics", {}),
        report_path="",
        result_path=str(scene_dir),
        simulation=({
            "record": archive.relative_to(_closed_loop_root()).as_posix() if archive else None,
            "path": str(archive) if archive else None,
            "scene_count": len(_archive_tokens(archive)) if archive else None,
            "vlm_model": status.get("model"),
        } if archive else None),
        raw=result,
    )

