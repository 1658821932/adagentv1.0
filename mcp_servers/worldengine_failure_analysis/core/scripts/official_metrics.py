"""Locate the official PDM score table for one simulation archive.

The analysis core runs inside the WorldEngine failure-analysis tree, so this
shim only makes the shared resolver importable there. All preference logic
lives in common/pdm_scores.py so every reader agrees on the same table.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pdm_scores import score_file, score_files  # noqa: E402  (path bootstrap)


def official_metrics_csv(root):
    """Path of the official PDM table this archive should be read from."""
    return score_file(root)
