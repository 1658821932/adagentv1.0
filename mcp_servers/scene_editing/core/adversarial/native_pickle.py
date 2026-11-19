"""Read scientific scene data with NumPy-version compatibility; never import data-named code."""

import collections
import pickle

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
