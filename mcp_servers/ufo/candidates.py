"""Reuse reconstruction SAM tracks as candidates for the Agent target selector.

Runs on CPU in ufoex. Evidence uses the same geometry/schema as DGGT discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def discover(workspace: Path, output: Path) -> dict:
    from PIL import Image, ImageDraw, ImageOps
    from .lane_edit import inventory
    from tools.tool05_instance_discovery.candidates import build_candidates
    from tools.tool05_instance_discovery.evidence import preprocess_image, write_object_cards

    workspace, output = workspace.resolve(), output.resolve()
    originals = [Image.open(workspace / f'input/rgb/video/0/{i:06d}.jpg').convert('RGB') for i in range(40)]
    frames = [preprocess_image(image) for image in originals]
    width, height = originals[0].size
    scaled_height = round(height * 518 / width / 14) * 14
    top = max(0, (scaled_height - 518) // 2)
    records = []
    for candidate in inventory(workspace):
        for row in candidate['observations']:
            x1, y1, x2, y2 = row['bbox']
            source_height, source_width = row['image_size']
            box = [math.floor(x1 * 518 / source_width), max(0, math.floor(y1 * scaled_height / source_height - top)),
                   math.ceil(x2 * 518 / source_width), min(frames[0].height, math.ceil(y2 * scaled_height / source_height - top))]
            if box[0] < box[2] and box[1] < box[3]:
                records.append({'track_id': candidate['track_id'], 'frame_index': row['frame'],
                                'bbox': box, 'class_id': row.get('class_id', 2), 'confidence': 1.0})
    instances, mapping = build_candidates(records, len(frames))
    for instance in instances:
        instance['sam_track_id'] = instance['track_id']
    evidence = output.parent / 'evidence'
    (evidence / 'frames').mkdir(parents=True, exist_ok=True)
    frame_images = []
    for index, frame in enumerate(frames):
        path = evidence / 'frames' / f'{index:03d}.jpg'
        frame.save(path, quality=95)
        frame_images.append(str(path))
    samples = sorted({round(39 * i / 5) for i in range(6)})
    sheet = Image.new('RGB', (1280, 900), '#111827')
    for slot, index in enumerate(samples):
        panel = ImageOps.contain(frames[index], (630, 270))
        draw = ImageDraw.Draw(panel)
        sx, sy = panel.width / frames[index].width, panel.height / frames[index].height
        for record in records:
            if record['frame_index'] == index:
                x1, y1, x2, y2 = record['bbox']
                draw.rectangle((x1*sx, y1*sy, x2*sx, y2*sy), outline='#00ffff', width=2)
                draw.text((x1*sx, max(0, y1*sy-12)), f"C{mapping[record['track_id']]}", fill='yellow')
        x, y = slot % 2 * 640, slot // 2 * 300
        sheet.paste(panel, (x, y+25))
        ImageDraw.Draw(sheet).text((x+5, y+5), f'frame {index}', fill='white')
    scene = evidence / 'candidates_scene.jpg'
    sheet.save(scene, quality=95)
    payload = {'schema_version': 1, 'selection_required': True,
               'source': {'backend': 'ufo', 'workspace': str(workspace), 'frame_count': 40,
                          'frame_size': list(frames[0].size), 'start_idx': 0,
                          'preprocessed_rgb_sha256': hashlib.sha256(b''.join(im.tobytes() for im in frames)).hexdigest()},
               'instances': instances, 'frame_images': frame_images,
               'evidence_images': [str(scene)], 'evidence_frame_indices': samples,
               'evidence_sha256': {str(scene): hashlib.sha256(scene.read_bytes()).hexdigest()},
               'object_cards': write_object_cards(instances, originals, evidence / 'object_cards')}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return {**payload, 'instance_info_path': str(output), 'candidate_count': len(instances)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    discover(args.workspace, args.output)
