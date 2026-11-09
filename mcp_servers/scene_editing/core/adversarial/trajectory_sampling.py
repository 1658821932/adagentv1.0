"""Keep requested states exact while avoiding nearly coincident numerical knots."""
import numpy as np


def construction_times(required,spatial_crossings,start,end):
    required=np.unique(np.asarray(required,dtype=float));kept=[]
    for value in sorted(spatial_crossings):
        if np.min(np.abs(required-value))<=.001 or (kept and value-kept[-1]<=.001):continue
        kept.append(value)
    mandatory=np.unique(np.r_[required,kept]);grid=np.arange(start,end,.025)
    grid=grid[np.min(np.abs(grid[:,None]-mandatory[None,:]),axis=1)>.001]
    return np.unique(np.r_[mandatory,grid])
