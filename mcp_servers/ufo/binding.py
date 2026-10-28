"""Bind a visually selected DGGT/YOLO vehicle to UFO's SAM track ID."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .actor_edit import match_sam_track


def bind_selected_track(workspace: Path, selection_path: Path) -> dict:
    import numpy as np
    from tools.tool05_instance_discovery.evidence import original_box
    from tools.tool05_instance_discovery.selection import load_bound_instances

    binding, selected = load_bound_instances(selection_path)
    candidates = json.loads(Path(binding['candidate_info_path']).read_text())
    if candidates.get('source', {}).get('backend') == 'ufo':
        if Path(candidates['source']['workspace']).resolve() != workspace.resolve():
            raise ValueError('Selected UFO candidate belongs to another reconstruction')
        ident = selected.get('sam_track_id')
        if type(ident) is not int or ident <= 0 or ident != selected.get('track_id'):
            raise ValueError('Invalid native SAM candidate binding')
        return {'selected_instance_id': binding['selected_instance_id'], 'sam_track_id': ident,
                'consensus': 1.0, 'binding_source': 'native SAM candidate snapshot'}
    manifest = json.loads((workspace / "input" / "input_manifest.json").read_text(encoding="utf-8"))
    height, width = manifest["image_size"]
    scene = workspace / "reconstruction" / "000" / "pred100k"
    votes = Counter()
    observations = []
    for record in selected["frames"]:
        index = int(record["frame_index"])
        source_path = scene / "sources" / f"{index:06d}.npz"
        if not source_path.is_file():
            continue
        x1, y1, x2, y2 = original_box(record["bbox"], (width, height))
        left = max(0, min(519, int(x1 * 520 / width)))
        top = max(0, min(351, int(y1 * 352 / height)))
        right = max(left + 1, min(520, int(x2 * 520 / width + .999)))
        bottom = max(top + 1, min(352, int(y2 * 352 / height + .999)))
        with np.load(source_path, allow_pickle=False) as source:
            try:
                track_id, dominance = match_sam_track(source, [(left, top, right, bottom)])
            except ValueError:
                continue
        if dominance < .7:
            continue
        votes[track_id] += 1
        observations.append({"frame": index, "sam_track_id": track_id,
                             "dominance": dominance, "ufo_box": [left, top, right, bottom]})
    if len(observations) < 3:
        raise ValueError("selected vehicle has fewer than 3 usable UFO/SAM matches")
    track_id, wins = votes.most_common(1)[0]
    consensus = wins / len(observations)
    if consensus < .7:
        raise ValueError(f"selected vehicle SAM track is inconsistent: consensus={consensus:.3f}")
    return {"selected_instance_id": binding["selected_instance_id"],
            "sam_track_id": track_id, "consensus": consensus,
            "observations": observations}
