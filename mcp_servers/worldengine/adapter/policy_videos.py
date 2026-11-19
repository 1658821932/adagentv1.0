"""Keep each policy run's videos independent of WorldEngine's legacy debug path."""
import hashlib
import json
from pathlib import Path
import shutil


def collect_policy_videos(result_dir, worldengine_root, started_at):
    folder = Path(result_dir) / 'videos'
    owned = sorted(folder.glob('*.mp4'))
    records = []
    if not owned:
        legacy = Path(worldengine_root) / 'projects/SimEngine/dbg/video_visualizer'
        sources = []
        for path in legacy.glob('*/*.mp4'):
            if path.stat().st_size > 0 and path.stat().st_mtime >= started_at - 2:
                sources.append(path)
        for path in sorted(sources, key=lambda p:p.stat().st_mtime)[:4]:
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / path.name
            # Once archived, an old run must never follow a later source overwrite.
            if destination.exists():
                raise FileExistsError('Policy video archive already exists: ' + str(destination))
            shutil.copy2(path, destination)
            records.append(dict(source=str(path), path=str(destination), publication='legacy_video_snapshot'))
        owned = sorted(folder.glob('*.mp4'))
    else:
        records = [dict(path=str(path), publication='run_owned_video') for path in owned]
    if owned:
        for record in records:
            path = Path(record['path'])
            record.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (folder/'manifest.json').write_text(json.dumps(dict(videos=records),ensure_ascii=False,indent=2))
    return [str(path) for path in owned]
