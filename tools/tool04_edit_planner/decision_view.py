"""Small coarse-planning view; original evidence remains available for audit."""
from collections import Counter
from copy import deepcopy
import math

MAX_NEARBY_NONVEHICLES = 8


def rounded(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def model_view(summary, evidence_path):
    """Retain all vehicle IDs and camera sheets without dumping all static boxes."""
    view=deepcopy(summary)
    actors=summary['actors']
    view['actors']=rounded([actor for actor in actors if actor['label']=='vehicle'])
    view['other_actor_counts']=dict(Counter(actor['label'] for actor in actors if actor['label']!='vehicle'))
    snapshots=[]
    for source in summary['snapshots']:
        row={key:rounded(value) for key,value in source.items() if key!='objects'}
        vehicles=[item for item in source['objects'] if item['label']=='vehicle']
        others=[item for item in source['objects'] if item['label']!='vehicle']
        def distance(item):
            box=item.get('box') or []
            return math.hypot(*box[:2]) if len(box)>=2 else float('inf')
        nearby=sorted(others,key=distance)[:MAX_NEARBY_NONVEHICLES]
        row['objects']=rounded(vehicles+nearby)
        row['other_actor_counts']=dict(Counter(item['label'] for item in others))
        row['omitted_nonvehicle_count']=len(others)-len(nearby)
        snapshots.append(row)
    view['snapshots']=snapshots
    evidence=summary['evidence']
    view['evidence']={key:deepcopy(evidence[key]) for key in (
        'annotation_path','eligible_vehicle_tracks','road_geometry_available',
        'missing_resources','added_vehicle_asset_available','review_scope') if key in evidence}
    view['evidence']['full_scene_evidence_path']=str(evidence_path)
    spatial=evidence.get('spatial_overview',{})
    view['evidence']['spatial_overview']={key:value for key,value in spatial.items() if key not in ('vehicle_catalog','trajectory')}
    if spatial.get('status')=='available':
        # Layout and ego route are supplied visually; individual time histories
        # can be queried instead of sending every box at multiple times.
        catalog={item['track_id']:item for item in spatial.get('vehicle_catalog',[])}
        view['actors']=[dict(actor,**{key:value for key,value in catalog.get(actor['track_id'],{}).items() if key!='track_id'}) for actor in view['actors']]
        view.pop('snapshots',None)
        view['ego_reference']=spatial.get('ego_summary')
    view['presentation']=dict(numeric_precision='Rounded to 2 decimal places for planning only; original precision is preserved in full evidence and tool05 inputs.',
        vehicle_catalog='All observed vehicle IDs retained; ordering is not a quality ranking.',
        nonvehicle_snapshots=f'Nearest {MAX_NEARBY_NONVEHICLES} nonvehicles per snapshot plus category counts; omission is not absence or proof of safety.')
    return view
