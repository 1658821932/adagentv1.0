"""Scene-level failure selection from closed-loop PDM average scores."""
from common.pdm_scores import read_scores


def load_scene_scores(rollout_dir):
    """Read whole-scene closed-loop PDM totals exported by WorldEngine.

    Returns (scores, csv_path) where scores maps the full scene token to its
    closed-loop PDM total score. React type is resolved centrally, so a
    non-reactive export is selected instead of reported as missing.
    """
    return read_scores(rollout_dir)


def failed_scene_anchors(scores, allowed_tokens, max_score=None):
    """Return requested scene anchors whose official PDM total is below threshold (default: <= 0.0)."""
    if max_score is None:
        failed = {token.split("-", 2)[-1] for token, score in scores.items() if score <= 0.0}
    else:
        failed = {token.split("-", 2)[-1] for token, score in scores.items() if score < float(max_score)}
    return failed & set(allowed_tokens)
