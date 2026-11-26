"""Read scientific scene data with NumPy-version compatibility; never import data-named code."""
from pathlib import Path
import json
import sys
import pickle
import numpy as np

from mcp_servers.scene_editing.core.adversarial.native_pickle import SceneDataUnpickler


def _load_one(path: Path) -> dict:
    with path.open("rb") as f:
        scenes = SceneDataUnpickler(f).load()
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError(f"Expected a non-empty WorldEngine scene dictionary in {path}")
    return scenes


def prepare(source, destination, limit=None):
    src_path = Path(source)
    if src_path.suffix.lower() == ".json":
        pkl_list = json.loads(src_path.read_text(encoding="utf-8"))
        if not isinstance(pkl_list, list) or not pkl_list:
            raise ValueError("Batch scenario manifest must be a non-empty JSON list of .pkl paths")
        scenes = {}
        for item in pkl_list:
            scenes.update(_load_one(Path(item)))
    else:
        scenes = _load_one(src_path)
    if limit is not None:
        scenes = dict(list(scenes.items())[:limit])
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(scenes, f, protocol=4)
    print("Prepared", len(scenes), "scene(s) for NumPy", np.__version__, flush=True)


if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
