"""Resolve official PDM score tables inside a simulation archive."""
from __future__ import annotations

from pathlib import Path

PREFIX = "all_scenes_pdm_averages"
SUFFIX_PREFERENCE = ("_R", "_NR")


def score_directory(rollout_dir) -> Path:
    base = Path(rollout_dir).expanduser()
    nested = base / "WE_output/openscene_format"
    return nested if nested.is_dir() else base


def score_files(rollout_dir) -> list[Path]:
    """List reactive, non-reactive and then any additional tables."""
    directory = score_directory(rollout_dir)
    priority = {PREFIX + suffix + '.csv': i for i, suffix in enumerate(SUFFIX_PREFERENCE)}
    return sorted(directory.glob(PREFIX + '_*.csv'), key=lambda path: (priority.get(path.name, len(priority)), path.name))


def score_file(rollout_dir, *, require_unique=False) -> Path:
    files = score_files(rollout_dir)
    directory = score_directory(rollout_dir)
    if not files:
        raise FileNotFoundError(f'No official PDM table ({PREFIX}_*.csv) in {directory}; run the closed-loop simulation first')
    if require_unique and len(files) != 1:
        names = ', '.join(path.name for path in files)
        raise ValueError(f'Multiple official PDM tables in {directory}: {names}. Provide a result directory with one official table.')
    return files[0]


def official_metrics_csv(root):
    """Path of the official PDM table this archive should be read from."""
    return score_file(root)
