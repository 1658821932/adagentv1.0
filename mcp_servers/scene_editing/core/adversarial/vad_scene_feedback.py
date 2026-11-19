"""Map native WE body states to the same fixed coordinates used by editing tools."""
import json
from pathlib import Path
import numpy as np


def feedback(editor,collision):
    path=Path(collision.get('states_path',''))
    if not path.is_file() or not getattr(editor,'frames',None):
        return dict(status='unavailable',reason='Native states or frozen origin unavailable; no coordinates inferred')
    from run_ego_reference import EGO_FRONT,EGO_REAR
    first=np.asarray(editor.frames[0]['ego2global'],float)
    angle=float(np.arctan2(first[1,0],first[0,0]))
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    # Native exporter translates global axes by the first rear axle. No UTM
    # translation or body-relative-left displacement should be added here.
    frames=[json.loads(line) for line in path.read_text().splitlines() if line]
    rows=[];target=editor.target['track_id'];offset=(EGO_FRONT-EGO_REAR)/2
    for frame in frames:
        own=frame['actors']['ego'];box=own['box'];yaw=float(np.arctan2(np.sin(box[4]-angle),np.cos(box[4]-angle)))
        center=np.asarray(box[:2])@rotation;rear=center-offset*np.array([np.cos(yaw),np.sin(yaw)])
        row=dict(step=frame['step'],time_s=float(frame['time_s']),ego_rear_axle_xy_m=rear.tolist(),
            ego_body_center_xy_m=center.tolist(),ego_heading_rad=yaw,
            ego_length_m=float(box[2]),ego_width_m=float(box[3]),
            ego_speed_mps=float(np.linalg.norm(own.get('velocity',[0.,0.]))))
        if target in frame['actors']:
            b=frame['actors'][target]['box'];point=np.asarray(b[:2])@rotation
            row.update(target_body_center_xy_m=point.tolist(),target_heading_rad=float(b[4]-angle),
                target_length_m=b[2],target_width_m=b[3])
        rows.append(row)
    closest=(collision.get('closest_approach') or {}).get('step')
    indices={0,len(rows)-1,min(range(len(rows)),key=lambda i:abs(rows[i]['time_s']-1.5))}
    if closest is not None:indices.update(i for i,row in enumerate(rows) if row['step']==closest)
    indices.update(np.linspace(0,len(rows)-1,6,dtype=int).tolist())
    # Keep takeover/closest/end plus up to3 extra observations in model context.
    priority=sorted(indices,key=lambda i:(i not in (0,len(rows)-1),rows[i]['step']!=closest,
        abs(rows[i]['time_s']-1.5),i))[:6]
    keyframes=[]
    for i in sorted(priority):
        row=rows[i];keyframes.append({key:([round(float(v),3) for v in value] if isinstance(value,list)
            else round(float(value),3) if isinstance(value,float) else value) for key,value in row.items()})
    return dict(status='measured',coordinate_frame='initial_ego',ego_reference_point='rear_axle',
        target_reference_point='body_center',keyframes=keyframes,full_precision_states=rows,
        native_to_scene_rotation_rad=-angle,
        interpretation='These are actual VAD states, not the recorded reference. Relative-left/yaw change alone does not establish avoidance, lane identity or a fixed-frame repositioning command. Compare paths on the diagram; unchanged observations do not predict another attack.')


def plot(editor,collision,view,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from run_ego_reference import EGO_FRONT,EGO_REAR,polygon
    rows=view['full_precision_states'];figure,(axis,gap_axis)=plt.subplots(1,2,figsize=(13,5))
    road=getattr(editor,'road',None)
    for geometry in getattr(road,'polygons',[]):
        axis.add_patch(Patch(np.asarray(geometry.exterior.coords),facecolor='#edf3e8',edgecolor='#bac8b0',linewidth=.5))
    yaw=np.asarray(editor.ego_yaw);reference=np.asarray(editor.ego)+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(yaw),np.sin(yaw)])
    axis.plot(reference[:,0],reference[:,1],'--',color='gray',label='recorded reference body centre')
    actual=np.array([row['ego_body_center_xy_m'] for row in rows]);axis.plot(actual[:,0],actual[:,1],'.-',label='actual VAD body centre')
    observed=[row for row in rows if 'target_body_center_xy_m' in row]
    if observed:
        points=np.array([row['target_body_center_xy_m'] for row in observed]);axis.plot(points[:,0],points[:,1],'.-',color='tab:red',label='attack target body centre')
    for row in view['keyframes']:
        axis.annotate(f"{row['time_s']:.1f}s",row['ego_body_center_xy_m'],fontsize=8)
    closest=(collision.get('closest_approach') or {}).get('step')
    moment=next((row for row in rows if row['step']==closest),rows[-1])
    axis.add_patch(Patch(np.asarray(polygon(*moment['ego_body_center_xy_m'],moment['ego_heading_rad'],
        EGO_FRONT+EGO_REAR,2.297).exterior.coords),fill=False,edgecolor='tab:blue'))
    if 'target_body_center_xy_m' in moment:
        axis.add_patch(Patch(np.asarray(polygon(*moment['target_body_center_xy_m'],moment['target_heading_rad'],
            moment['target_length_m'],moment['target_width_m']).exterior.coords),fill=False,edgecolor='tab:red'))
    # Avoid stretching the figure to distant road-map polygons.
    extent=np.vstack([reference,actual,*([points] if observed else [])]);axis.set_xlim(extent[:,0].min()-8,extent[:,0].max()+8)
    axis.set_ylim(extent[:,1].min()-8,extent[:,1].max()+8);axis.set_aspect('equal');axis.legend(fontsize=8);axis.grid(alpha=.2)
    axis.set(xlabel='initial ego forward x (m)',ylabel='initial ego left y (m)',title='Same fixed coordinates as vehicle editing')
    from box_geometry import body_contact
    from run_ego_reference import EGO_WIDTH
    times=[];gaps=[];speeds=[]
    for row in observed:
        box=[*row['ego_body_center_xy_m'],EGO_FRONT+EGO_REAR,EGO_WIDTH,row['ego_heading_rad']]
        target=[*row['target_body_center_xy_m'],row['target_length_m'],row['target_width_m'],row['target_heading_rad']]
        times.append(row['time_s']);gaps.append(body_contact(box,target)[1]);speeds.append(row['ego_speed_mps'])
    gap_axis.plot(times,gaps,label='actual body gap (m)',color='tab:red');twin=gap_axis.twinx()
    twin.plot(times,speeds,label='actual speed (m/s)',color='tab:blue');twin.set_ylabel('actual speed (m/s)')
    gap_axis.axvspan(0,1.5,color='gray',alpha=.15);gap_axis.set(xlabel='scene time (s)',ylabel='body gap (m)',title='Actual motion; shaded WE history')
    gap_axis.grid(alpha=.2);figure.suptitle(f"VAD target contact={collision['target_collision_detected']} | sampled geometry, not causal avoidance attribution")
    figure.tight_layout();path=Path(folder)/'vadv2-feedback.png';figure.savefig(path,dpi=120);plt.close(figure)
    return path
