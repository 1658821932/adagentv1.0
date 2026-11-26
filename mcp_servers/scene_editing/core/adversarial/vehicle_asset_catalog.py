"""Load vehicle assets with verified class and native dimensions from the project catalog."""
import json
from math import isfinite
from pathlib import Path


def load_vehicle_assets(vehicle_class, directory=None):
    root = (Path(directory) if directory is not None
            else Path(__file__).resolve().parents[4] / 'data/assets').resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    assets = []
    ids = set()
    for item in manifest['assets']:
        if vehicle_class != 'any' and vehicle_class not in item['vehicle_classes']:
            continue
        asset_id = item['asset_id']
        path = (root / item['asset_path']).resolve()
        size = item['size_m']
        if (not asset_id or asset_id in ids or root not in path.parents
                or path.suffix.lower() != '.ply' or not path.is_file()
                or len(size) != 3
                or any(type(value) not in (int, float) or not isfinite(value) or value <= 0
                       for value in size)):
            raise ValueError(f'Invalid vehicle catalog entry: {asset_id!r}')
        ids.add(asset_id)
        assets.append(dict(asset_id=asset_id, path=str(path), size_m=size))
    if not assets:
        raise ValueError(f"No catalog assets have a verified {vehicle_class!r} class")
    return assets
