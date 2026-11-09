"""Bind full lane topology to the explicitly located official-map polygons."""
from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate


def align(features,official):
    angle=float(official['first_frame_yaw_rad'])
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    pairs=[];shifts=[]
    for item in official['features'].values():
        source=features.get(str(item.get('token')))
        if not source or 'polygon' not in source:continue
        expected=Polygon(np.asarray(item['polygon'])[:,:2]@rotation.T)
        original=Polygon(np.asarray(source['polygon'])[:,:2])
        if not expected.is_valid or not original.is_valid or expected.area<1e-6:continue
        if abs(expected.area-original.area)>max(1e-6,expected.area*1e-6):continue
        shift=np.array(expected.centroid.coords[0])-np.array(original.centroid.coords[0])
        pairs.append((expected,original));shifts.append(shift)
    if len(pairs)<2:raise ValueError('Full lane map needs at least two matching official polygons to establish its translation')
    offset=np.median(np.array(shifts),axis=0)
    error=max(a.hausdorff_distance(translate(b,xoff=offset[0],yoff=offset[1])) for a,b in pairs)
    if error>.001:raise ValueError(f'Lane map coordinate binding differs from official polygons by {error:.6f}m')
    result=deepcopy(features)
    for node in result.values():
        for field in ('polyline','polygon'):
            if field in node:
                points=np.asarray(node[field],float).copy();points[:,:2]+=offset;node[field]=points
        if node.get('type') in ('LANE_SURFACE_STREET','LANE_FREEWAY') and ('polyline' not in node or 'roadblock_id' not in node):
            raise ValueError('Source lane is missing its centreline/topology metadata')
    return result,dict(matched_polygon_count=len(pairs),translation_to_native_m=offset.tolist(),max_polygon_hausdorff_error_m=float(error),method='Shared polygon IDs/shape establish the source map translation; no synthetic lanes')
