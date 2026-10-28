"""Independent before/after asset audit for a completed inserted-object edit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def audit_pair(baseline_result: Path, edited_result: Path, object_id: str, output: Path) -> dict:
    import numpy as np
    import torch
    from .metric_assets import FIELDS
    before,after=[json.loads(p.read_text()) for p in (baseline_result,edited_result)]
    if not before.get('render_complete') or not after.get('render_complete'):
        raise ValueError('Both cases must have completed native rendering')
    if before['frame_count']!=after['frame_count'] or before['reference_frame_index']!=after['reference_frame_index']:
        raise ValueError('Review requires the same recorded frames and fixed reference frame')
    basis=np.asarray(after['road_basis_flu_world'])
    if not np.allclose(basis,before['road_basis_flu_world'],atol=1e-10):
        raise ValueError('Baseline and edited road bases differ')
    manifests=[json.loads(Path(case['asset_manifest']).read_text()) for case in (before,after)]
    if manifests[0]['metric_calibration']['sha256']!=manifests[1]['metric_calibration']['sha256']:
        raise ValueError('Baseline and edited calibrations differ')
    checkpoints=[torch.load(m['asset_ckpt'],map_location='cpu',weights_only=False) for m in manifests]
    bg0,bg1=[c['background']['state_dict'] for c in checkpoints]
    if set(bg0)!=set(bg1) or any(not torch.equal(bg0[key],bg1[key]) for key in bg0):
        raise ValueError('Before/after background tensors differ; cannot isolate this translation')
    directories=[Path(case['asset_manifest']).parent for case in (before,after)]
    if (directories[0]/'cameras.npz').read_bytes()!=(directories[1]/'cameras.npz').read_bytes():
        raise ValueError('Before/after camera files differ')
    inserted=object_id!='selected'
    if inserted and (set(after['inserted_objects'])!={object_id} or set(before['inserted_objects'])!={object_id}):
        raise ValueError('Inserted-object audit requires exactly one matching inserted object')
    if not inserted and (after['inserted_objects'] or before['inserted_objects']):
        raise ValueError('Selected-original audit does not mix additional insertions')
    count=after['inserted_objects'][object_id]['gaussian_count'] if inserted else None
    scale=after['meters_per_native_unit']
    dimensions=np.asarray(after['inserted_objects'][object_id]['dimensions_m']) if inserted else None
    calibration=json.loads(Path(after['calibration_path']).read_text())
    rows=[];max_error=max_size_error=0.
    for index in range(after['frame_count']):
        a,b=[torch.load(folder/'dynamic'/f'{index:06d}.pt',map_location='cpu',weights_only=True) for folder in directories]
        if len(a['means'])!=len(b['means']):raise ValueError('Actor counts differ')
        if inserted:
            selected=np.arange(len(a['means'])-count,len(a['means']))
        else:
            from .actor_edit import actor_mask
            track=after['target_binding']['sam_track_id']
            if track!=before['target_binding']['sam_track_id']:raise ValueError('Selected SAM identities differ')
            workspace=Path(calibration['reconstruction']['workspace'])
            with np.load(workspace/f'reconstruction/000/pred100k/sources/{index:06d}.npz',allow_pickle=False) as source:
                mask=actor_mask(source);selected=np.flatnonzero(source['tracks'][mask]==track)
            if not len(selected):raise ValueError('Selected actor absent from an independent review frame')
        other=np.ones(len(a['means']),dtype=bool);other[selected]=False
        for key in FIELDS:
            if not torch.equal(a[key][other],b[key][other]):
                raise ValueError(f'Non-target actor field changed: frame={index} key={key}')
            if key!='means' and not torch.equal(a[key],b[key]):
                raise ValueError(f'Translation changed a shape/appearance field: frame={index} key={key}')
        delta=(b['means'][selected].double()-a['means'][selected].double()).numpy()@basis*scale
        operation=next(op for op in after['frames'][index]['operations'] if op.get('operation')=='translate' and op['target_id']==object_id)
        requested=np.asarray(operation['requested_offset_m'])
        error=float(np.max(abs(delta-requested)));max_error=max(max_error,error)
        # This review case uses yaw=0. Reject a size mismatch rather than silently
        # treating a world-aligned bbox of a rotated object as physical size.
        local=b['means'][selected].double().numpy()@basis*scale
        expected_size=dimensions if inserted else np.ptp(a['means'][selected].double().numpy()@basis*scale,axis=0)
        size_error=float(np.max(abs(np.ptp(local,axis=0)-expected_size)));max_size_error=max(max_size_error,size_error)
        if error>.001 or size_error>.001:raise ValueError(f'Independent metric/shape verification failed at frame {index}')
        mean=delta.mean(0)
        rows.append({'frame':index,'timestamp_seconds':after['frames'][index]['timestamp_seconds'],
                     'requested_right_m':-float(requested[1]),'actual_right_m':-float(mean[1]),
                     'actual_forward_m':float(mean[0]),'actual_up_m':float(mean[2]),
                     'max_point_error_m':error,'max_dimension_error_m':size_error})
    output.mkdir(parents=True,exist_ok=True)
    csv_path=output/'independent_distance_audit.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'status':'passed_coordinate_execution_checks','object_id':object_id,'frames':len(rows),
            'final_right_displacement_m':rows[-1]['actual_right_m'],'max_point_error_m':max_error,
            'max_dimension_error_m':max_size_error,'background_equal_between_cases':True,
            'original_non_targets_equal_between_cases':True,'shape_and_appearance_equal_between_cases':True,
            'camera_bytes_equal':True,'dimensions_m':dimensions.tolist() if inserted else None,
            'dimension_source':after['inserted_objects'][object_id].get('dimension_source') if inserted else 'observed Gaussian extent, no canonical-size claim',
            'temporal_shape_consistency_certified':False,
            'calibration_validation_scope':after.get('calibration_validation_scope'),
            'absolute_accuracy_certified':False,'visual_status':'requires_human_review',
            'baseline_result':str(baseline_result.resolve()),'edited_result':str(edited_result.resolve()),
            'csv_path':str(csv_path.resolve()),'road_geometry_repair':after.get('road_geometry_repair'),
            'note':'Measured from serialized before/after tensors, separately from execution audit. Coordinate accuracy does not certify real-world scale or road geometry.'}
    (output/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    width,height=760,300;left,top=65,35;plotw,ploth=660,210
    def points(field):
        return ' '.join(f'{left+plotw*r["timestamp_seconds"]/max(rows[-1]["timestamp_seconds"],1e-6):.2f},{top+ploth*(1-r[field]/3.2):.2f}' for r in rows)
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>'
         f'<text x="65" y="20" font-family="sans-serif" font-size="15">Right displacement in estimated coordinate meters (not surveyed distance)</text>'
         f'<path d="M{left},{top} V{top+ploth} H{left+plotw}" fill="none" stroke="black"/>'
         f'<polyline points="{points("requested_right_m")}" fill="none" stroke="#93c5fd" stroke-width="6"/>'
         f'<polyline points="{points("actual_right_m")}" fill="none" stroke="#166534" stroke-width="2"/>'
         f'<text x="65" y="280" font-family="sans-serif" font-size="13">Blue: requested; green: independently measured. Max point error: {max_error:.3g} m</text></svg>')
    (output/'distance_curve.svg').write_text(svg)
    return {key:value for key,value in report.items() if key!='road_geometry_repair'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-result',type=Path,required=True)
    parser.add_argument('--edited-result',type=Path,required=True)
    parser.add_argument('--object-id',default='review_car')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit_pair(args.baseline_result,args.edited_result,args.object_id,args.output),allow_nan=False))
