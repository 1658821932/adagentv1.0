"""Scene-level failure selection from closed-loop PDM average scores."""
import csv
from pathlib import Path


def load_scene_scores(rollout_dir):
    """Read whole-scene closed-loop PDM totals exported by WorldEngine.

    Returns (scores, csv_path) where scores maps the full scene token to its
    closed-loop PDM total score.
    """
    files = sorted(Path(rollout_dir).glob("all_scenes_pdm_averages_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No all_scenes_pdm_averages_*.csv in {rollout_dir}; "
            "run the closed-loop simulation first")
    if len(files) > 1:
        raise ValueError("Multiple PDM average files: " + ", ".join(f.name for f in files))
    scores = {}
    with files[0].open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = (row.get("token") or "").strip()
            if not token or token == "overall_average":
                continue
            try:
                scores[token] = float(row["score"])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid PDM score for scene {token}") from None
    if not scores:
        raise ValueError(f"No scene scores in {files[0]}")
    return scores, files[0]


def failed_scene_anchors(scores, allowed_tokens):
    """Anchors (log-name suffixes) of requested scenes with a zero PDM score."""
    failed = {token.split("-", 2)[-1] for token, score in scores.items() if score <= 0.0}
    return failed & set(allowed_tokens)
