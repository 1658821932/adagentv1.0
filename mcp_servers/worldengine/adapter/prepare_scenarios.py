"""Read scientific scene data with NumPy-version compatibility; never import data-named code."""
from pathlib import Path
import sys
import pickle
import collections
import numpy as np

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
    ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
    ("numpy._core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
    ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "set"): set, ("builtins", "slice"): slice,
}
class SceneDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError("Unsupported scene data global: " + module + "." + name)
    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent scene objects are not supported")

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
