"""Explicit ray-preserving planar ROAD proxy for editable Gaussian assets.

The native learned road can lie in front of an inserted car despite its metric
pose, causing false occlusion. This opt-in repair projects semantic static-road
Gaussians onto the declared local plane, preserving each source-view center and
footprint. Buildings, vegetation, existing actors and their appearance stay out
of scope. This is a planar scene approximation, not measured road geometry.
"""
from __future__ import annotations

from pathlib import Path


def repair_road(state: dict, workspace: Path, poses, intrinsics, times, plane: dict) -> dict:
    import numpy as np
    means = state['gauss_params.means'].numpy()
    log_scales = state['gauss_params.scales'].numpy()
    logits = state['gauss_params.opacities'].numpy().reshape(-1)
    source_times = state['source_time_s'].numpy().reshape(-1)
    times = np.asarray(times,dtype=np.float64)
    if len(source_times)!=len(means) or log_scales.shape!=means.shape:
        raise ValueError('Road proxy needs matching native temporal Gaussian attributes')
    right=np.clip(np.searchsorted(times,source_times),0,len(times)-1)
    left=np.maximum(0,right-1)
    frame_ids=np.where(abs(source_times-times[left])<=abs(source_times-times[right]),left,right)
    if np.max(abs(source_times-times[frame_ids]))>.001:
        raise ValueError('Cannot associate road Gaussians with source camera timestamps')
    normal=np.asarray(plane['normal_world'],dtype=np.float64)
    offset=float(plane['offset_native'])
    changed=removed=0
    max_error=0.
    frame_rows=[]
    for frame in np.unique(frame_ids):
        indices=np.flatnonzero(frame_ids==frame)
        old=means[indices].astype(np.float64)
        pose=np.asarray(poses[frame],dtype=np.float64)
        K=np.asarray(intrinsics[frame],dtype=np.float64)
        xyz=(old-pose[:3,3])@pose[:3,:3]
        pix=xyz@K.T
        u=pix[:,0]/np.maximum(pix[:,2],1e-12);v=pix[:,1]/np.maximum(pix[:,2],1e-12)
        with np.load(workspace/f'reconstruction/000/pred100k/sources/{frame:06d}.npz',allow_pickle=False) as source:
            h,w=source['classes'].shape
            visible=np.isfinite(xyz).all(1)&(xyz[:,2]>.2)&(u>=0)&(u<w)&(v>=0)&(v<h)
            locations=np.flatnonzero(visible)
            x=np.floor(u[locations]).astype(int);y=np.floor(v[locations]).astype(int)
            road=(source['classes'][y,x]==0)&~source['dynamic'][y,x]&(source['coverage'][y,x]<.01)
            selected=locations[road]
        if not len(selected):continue
        rays=old[selected]-pose[:3,3]
        denominator=rays@normal
        ratio=np.full(len(selected),np.nan)
        np.divide(-(float(pose[:3,3]@normal)+offset),denominator,out=ratio,where=abs(denominator)>1e-9)
        z=xyz[selected,2]*ratio
        good=np.isfinite(ratio)&(ratio>0)&(z>.2)&(z<400)
        global_ids=indices[selected[good]]
        new=pose[:3,3]+rays[good]*ratio[good,None]
        means[global_ids]=new.astype(means.dtype)
        log_scales[global_ids]+=np.log(ratio[good,None]).astype(log_scales.dtype)
        invalid=indices[selected[~good]]
        logits[invalid]=-np.inf  # Exactly zero opacity for nonphysical plane intersections.
        if len(global_ids):
            projected=((means[global_ids].astype(np.float64)-pose[:3,3])@pose[:3,:3])@K.T
            uv=projected[:,:2]/projected[:,2,None]
            error=float(np.max(abs(uv-np.column_stack([u[selected[good]],v[selected[good]]]))))
            max_error=max(max_error,error)
            if error>.02:
                raise ValueError(f'Road proxy source projection error exceeds .02 pixels: {error}')
        changed+=len(global_ids);removed+=len(invalid)
        frame_rows.append({'frame':int(frame),'projected_to_plane':len(global_ids),'invalid_road_intersections_removed':len(invalid)})
    if changed==0:
        raise ValueError('No semantic static road Gaussians available for the requested proxy')
    return {'mode':'planar_proxy','road_gaussians_repositioned':changed,
            'invalid_road_intersections_removed':removed,'max_source_projection_error_pixels':max_error,
            'source_view_footprint_preserved':True,'other_semantic_classes_unchanged':True,
            'scope':'estimated local planar road approximation; changes static road geometry, not a dense reconstruction accuracy certificate',
            'frames':frame_rows}
