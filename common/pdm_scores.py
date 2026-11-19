"""Official PDM score tables: one resolver shared by every reader.

Tools used to hardcode a single filename each, so a run exported with a
different react type looked like a missing result. The tables differ only by
suffix; reactive (``_R``) is the official closed-loop baseline and non-reactive
(``_NR``) is the fallback, so readers never fail over naming.
"""
from __future__ import annotations

import csv
from pathlib import Path

PREFIX = "all_scenes_pdm_averages"
SUFFIX_PREFERENCE = ("_R", "_NR")


def score_directory(rollout_dir) -> Path:
    """Locate the openscene_format directory that holds the official tables."""
    base = Path(rollout_dir).expanduser()
    if base.name == "openscene_format" and base.is_dir():
        return base
    nested = base / "WE_output/openscene_format"
    return nested if nested.is_dir() else base


def score_files(rollout_dir) -> list[Path]:
    """Every official table found, reactive first, then non-reactive, then others."""
    directory = score_directory(rollout_dir)
    found = {path.name: path for path in directory.glob(PREFIX + "_*.csv")}
    ordered: list[Path] = []
    for suffix in SUFFIX_PREFERENCE:
        path = found.pop(PREFIX + suffix + ".csv", None)
        if path is not None:
            ordered.append(path)
    ordered.extend(found[name] for name in sorted(found))
    return ordered


def score_file(rollout_dir, *, require_unique=False) -> Path:
    """The single table a reader should use for this rollout."""
    files = score_files(rollout_dir)
    if not files:
        raise FileNotFoundError(f'No official PDM table ({PREFIX}_*.csv) in {score_directory(rollout_dir)}; '
                                'run the closed-loop simulation first')
    if require_unique and len(files) != 1:
        names = ', '.join(path.name for path in files)
        raise ValueError(f'Multiple official PDM tables in {score_directory(rollout_dir)}: {names}. '
                         'Provide a result directory with one official table.')
    return files[0]


def read_scores(rollout_dir) -> tuple[dict[str, float], Path]:
    """Scene-token to official PDM total, read from the resolved table."""
    path = score_file(rollout_dir)
    scores: dict[str, float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = (row.get("token") or "").strip()
            if not token or token == "overall_average":
                continue
            try:
                scores[token] = float(row["score"])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid PDM score for scene {token}") from None
    if not scores:
        raise ValueError(f"No scene scores in {path}")
    return scores, path
