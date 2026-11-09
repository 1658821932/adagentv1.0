"""Validate target replay coverage and alignment before rendering any frame."""
import numpy as np


def assess_replay(frame_ids, timestamps, annotated_xy, source_times, source_positions, limit_m=.5):
    annotated_xy=np.asarray(annotated_xy,float)
    if not frame_ids or len(frame_ids)!=len(timestamps) or annotated_xy.shape!=(len(frame_ids),2):
        raise ValueError('Every requested frame needs one timestamp and one annotation')
    if not np.isfinite(annotated_xy).all():
        raise ValueError('Asset annotations must be finite')
    positions=np.asarray(source_positions,float)
    if not np.isfinite(positions).all():
        raise ValueError('Asset source poses must be finite')
    static=positions.shape==(3,)
    times=np.asarray(source_times,dtype=np.int64) if not static else None
    if not static and (positions.ndim!=2 or positions.shape!=(len(times),3) or len(times)<2 or np.any(np.diff(times)<=0)):
        raise ValueError('Dynamic asset poses need matching, strictly increasing timestamps')
    rows=[]
    for index,timestamp,annotation in zip(frame_ids,timestamps,annotated_xy):
        covered=static or bool(times[0]<=timestamp<=times[-1])
        row=dict(frame_index=int(index),timestamp=int(timestamp),time_covered=covered)
        if not covered:
            row.update(passed=False,reason='outside_asset_replay_coverage',residual_m=None)
        else:
            if static:
                position=positions
            else:
                relative=times-times[0]
                position=np.array([np.interp(timestamp-int(times[0]),relative,positions[:,axis]) for axis in range(3)])
            residual=float(np.linalg.norm(position[:2]-annotation))
            passed=bool(np.isfinite(residual) and residual<=limit_m)
            row.update(passed=passed,reason=None if passed else 'pose_annotation_mismatch',residual_m=residual)
        rows.append(row)
    return dict(status='passed' if rows and all(row['passed'] for row in rows) else 'failed',
                limit_m=limit_m,frames=rows,source_pose_interpolation='relative-time linear; no extrapolation',
                scope='Source target asset coverage/alignment only; not visual or physical acceptance')
