"""Read scientific scene data with NumPy-version compatibility; never import data-named code."""
from pathlib import Path
import sys
import pickle
import numpy as np

from mcp_servers.scene_editing.core.adversarial.native_pickle import SceneDataUnpickler


def prepare(source, destination, limit=None):
    with Path(source).open("rb") as f:
        scenes = SceneDataUnpickler(f).load()
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("Expected a non-empty WorldEngine scene dictionary")
    if limit is not None:
        scenes = dict(list(scenes.items())[:limit])
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(scenes, f, protocol=4)
    print("Prepared", len(scenes), "scene(s) for NumPy", np.__version__, flush=True)

if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else None)
