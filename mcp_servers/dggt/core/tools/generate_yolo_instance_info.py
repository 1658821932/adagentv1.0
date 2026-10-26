#!/usr/bin/env python3
"""Generate a DGGT instance_info.json from one YOLO + ByteTrack sequence."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from datasets.dataset import WaymoOpenDataset


def select_primary_track(records):
    """Return the track id with the largest accumulated bbox area."""
    if not records:
        raise ValueError("No car track was detected in the sequence")

    areas = defaultdict(float)
    for record in records:
        x1, y1, x2, y2 = record["bbox"]
        areas[int(record["track_id"])] += (x2 - x1) * (y2 - y1)

    # Prefer the smaller raw id for deterministic results if areas tie.
    return max(areas, key=lambda track_id: (areas[track_id], -track_id))


def _frame_to_pil(frame):
    """Convert one [3, H, W] RGB tensor in [0, 1] to a PIL RGB image."""
    array = (
        frame.detach()
        .to(device="cpu")
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _integer_bbox(values, width, height):
    x1, y1, x2, y2 = values
    bbox = [
        max(0, min(width, math.floor(float(x1)))),
        max(0, min(height, math.floor(float(y1)))),
        max(0, min(width, math.ceil(float(x2)))),
        max(0, min(height, math.ceil(float(y2)))),
    ]
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return None
    return bbox


def _result_records(result, frame_index, width, height):
    boxes = result.boxes
    if boxes.id is None:
        return []

    coordinates = boxes.xyxy.detach().cpu().tolist()
    track_ids = boxes.id.detach().cpu().tolist()
    records = []
    for coordinates_one, track_id in zip(coordinates, track_ids):
        bbox = _integer_bbox(coordinates_one, width, height)
        if bbox is not None:
            records.append({
                "track_id": int(track_id),
                "frame_index": frame_index,
                "bbox": bbox,
            })
    return records


def _write_visualizations(frames, selected_records, vis_dir):
    vis_dir.mkdir(parents=True, exist_ok=True)
    records_by_frame = {record["frame_index"]: record for record in selected_records}
    for frame_index, image in enumerate(frames):
        canvas = image.copy()
        record = records_by_frame.get(frame_index)
        if record is not None:
            draw = ImageDraw.Draw(canvas)
            x1, y1, x2, y2 = record["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text(
                (x1, max(0, y1 - 15)),
                f"instance 0 track {record['track_id']}",
                fill="red",
            )
        canvas.save(vis_dir / f"{frame_index:04d}.png")


def generate_instance_info(args):
    """Run preprocessing, tracking, primary-track selection, and JSON output."""
    dataset = WaymoOpenDataset(
        args.image_dir,
        scene_names=[args.scene_name],
        sequence_length=args.sequence_length,
        start_idx=args.start_idx,
        mode=2,
        views=1,
    )
    sample = dataset[0]
    images = sample["images"]

    # Import at runtime so importing this module for tests does not require YOLO.
    from ultralytics import YOLO

    model = YOLO(args.weights)
    frames = []
    records = []
    for frame_index, frame in enumerate(images):
        image = _frame_to_pil(frame)
        frames.append(image)
        result = model.track(
            source=image,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2],
            conf=args.conf,
            verbose=False,
        )[0]
        records.extend(
            _result_records(result, frame_index, image.width, image.height)
        )

    primary_track_id = select_primary_track(records)
    selected_records = sorted(
        (record for record in records if record["track_id"] == primary_track_id),
        key=lambda record: record["frame_index"],
    )
    output = {
        "instances": [{
            "instance_id": 0,
            "frames": [
                {
                    "frame_index": record["frame_index"],
                    "bbox": record["bbox"],
                }
                for record in selected_records
            ],
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if args.vis_dir is not None:
        _write_visualizations(frames, selected_records, args.vis_dir)

    return primary_track_id, selected_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one primary car instance_info.json for DGGT Mode2."
    )
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--weights", default="yolo11m.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vis_dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    track_id, selected_records = generate_instance_info(args)
    print(
        f"Selected ByteTrack track_id={track_id}; "
        f"frames={len(selected_records)}; output={args.output}"
    )
    if args.vis_dir is not None:
        print(f"Visualization directory: {args.vis_dir}")


if __name__ == "__main__":
    main()
