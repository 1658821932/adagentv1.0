"""Detect all car tracks using the deployed DGGT preprocessing and detector.

Runs in the DGGT environment. No changes to the shared DGGT checkout are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tools.tool05_instance_discovery.evidence import (
    preprocess_image,
    write_object_cards,
)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_frames(image_dir, scene_name, video_path, count, start_idx):
    from PIL import Image

    if video_path:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        originals = []
        try:
            for _ in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"video has fewer than {count} requested frames")
                originals.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
        finally:
            cap.release()
    else:
        paths = sorted(
            p
            for p in (Path(image_dir) / scene_name / "images").iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        front = [p for p in paths if p.stem.endswith("_0")]
        if front:
            paths = front
        paths = paths[start_idx : start_idx + count]
        if len(paths) != count:
            raise ValueError(f"image sequence has fewer than {count} requested frames")
        originals = [Image.open(p).convert("RGB") for p in paths]
    return originals, [preprocess_image(image) for image in originals]


def records_from_result(result, index, width, height):
    import math

    boxes = result.boxes
    if boxes.id is None:
        return []
    records = []
    for xyxy, track_id, category, confidence in zip(
        boxes.xyxy.cpu().tolist(),
        boxes.id.cpu().tolist(),
        boxes.cls.cpu().tolist(),
        boxes.conf.cpu().tolist(),
    ):
        x1, y1, x2, y2 = xyxy
        bbox = [
            max(0, min(width, math.floor(x1))),
            max(0, min(height, math.floor(y1))),
            max(0, min(width, math.ceil(x2))),
            max(0, min(height, math.ceil(y2))),
        ]
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            records.append(
                {
                    "track_id": int(track_id),
                    "frame_index": index,
                    "bbox": bbox,
                    "class_id": int(category),
                    "confidence": float(confidence),
                }
            )
    return records


def build_candidates(records, frame_count):
    """Preserve every detected track with a deterministic scene-local ID."""
    if frame_count <= 0:
        raise ValueError("No input frames")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["track_id"]].append(record)
    instances = []
    mapping = {}
    for instance_id, (track_id, track) in enumerate(sorted(grouped.items())):
        track = sorted(track, key=lambda item: item["frame_index"])
        mapping[track_id] = instance_id
        indices = [item["frame_index"] for item in track]
        gaps = [b - a - 1 for a, b in zip(indices, indices[1:])]
        area = sum(
            (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]) for r in track
        )
        instances.append(
            {
                "instance_id": instance_id,
                "track_id": track_id,
                "class": VEHICLE_CLASSES.get(track[0].get("class_id", 2), "vehicle"),
                "detector_class_id": track[0].get("class_id", 2),
                "frames": [
                    {"frame_index": r["frame_index"], "bbox": r["bbox"]} for r in track
                ],
                "summary": {
                    "visible_frames": len(track),
                    "first_frame": indices[0],
                    "last_frame": indices[-1],
                    "max_missing_gap": max(gaps, default=0),
                    "coverage": len(track) / frame_count,
                    "mean_confidence": sum(r.get("confidence", 1.0) for r in track)
                    / len(track),
                    "accumulated_bbox_area": area,
                    "warnings": (["short_track"] if len(track) < 3 else [])
                    + (["tracking_gap"] if max(gaps, default=0) > 2 else []),
                },
            }
        )
    return instances, mapping


def main(argv=None):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image_dir")
    source.add_argument("--video_path")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--sequence_length", type=int, default=40)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--weights", default="yolo11m.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vis_dir", type=Path)
    args = parser.parse_args(argv)
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from ultralytics import YOLO

    originals, frames = load_frames(
        args.image_dir,
        args.scene_name,
        args.video_path,
        args.sequence_length,
        args.start_idx,
    )
    model = YOLO(args.weights)
    records = []
    for index, image in enumerate(frames):
        result = model.track(
            source=image,
            persist=True,
            agnostic_nms=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES),
            conf=args.conf,
            verbose=False,
        )[0]
        records.extend(records_from_result(result, index, image.width, image.height))
    instances, mapping = build_candidates(records, len(frames))
    vis = args.vis_dir or args.output.parent / (args.output.stem + "_evidence")
    vis.mkdir(parents=True, exist_ok=True)
    frame_dir = vis / "frames"
    frame_dir.mkdir(exist_ok=True)
    frame_images = []
    for index, image in enumerate(frames):
        path = frame_dir / f"{index:03d}.jpg"
        image.save(path, quality=95)
        frame_images.append(str(path.resolve()))
    # Include all candidates in the evidence, not only the largest ones.
    scene = Image.new("RGB", (1280, 810), "#161b22")
    draw = ImageDraw.Draw(scene)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), 16)
        if font_path.is_file()
        else ImageFont.load_default()
    )
    evidence_indices = sorted({round((len(frames) - 1) * i / 5) for i in range(6)})
    for slot, index in enumerate(evidence_indices):
        source = frames[index]
        image = ImageOps.contain(source, (630, 240))
        sx, sy = image.width / source.width, image.height / source.height
        overlay = ImageDraw.Draw(image)
        for record in records:
            if record["frame_index"] != index:
                continue
            x1, y1, x2, y2 = record["bbox"]
            box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            overlay.rectangle(box, outline="#00ff88", width=2)
            x, y = box[:2]
            label = f"C{mapping[record['track_id']]}"
            overlay.rectangle(
                (x, max(0, y - 18), x + 38, max(0, y - 18) + 17), fill="black"
            )
            overlay.text((x + 2, max(0, y - 18)), label, fill="#00ff88", font=font)
        x, y = (slot % 2) * 640, (slot // 2) * 270
        scene.paste(image, (x, y + 24))
        draw.text((x + 8, y + 6), f"frame {index}", fill="white", font=font)
    scene_path = vis / "candidates_scene.jpg"
    scene.save(scene_path, quality=93)
    crop_paths = []
    for start in range(0, len(instances), 12):
        chunk = instances[start : start + 12]
        sheet = Image.new("RGB", (1200, 180 * ((len(chunk) + 3) // 4)), "#161b22")
        for slot, instance in enumerate(chunk):
            track = instance["frames"]
            record = max(
                track,
                key=lambda r: (
                    (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])
                ),
            )
            instance["summary"]["crop_frame_index"] = record["frame_index"]
            crop = ImageOps.contain(
                frames[record["frame_index"]].crop(record["bbox"]), (290, 140)
            )
            x, y = (slot % 4) * 300, (slot // 4) * 180
            sheet.paste(crop, (x, y + 30))
            ImageDraw.Draw(sheet).text(
                (x + 4, y + 6),
                f"C{instance['instance_id']} frame {record['frame_index']} visible {len(track)}/{len(frames)}",
                fill="white",
                font=font,
            )
        path = vis / f"candidate_crops_{start // 12:02d}.jpg"
        sheet.save(path, quality=93)
        crop_paths.append(str(path.resolve()))
    car_areas = {
        i["track_id"]: i["summary"]["accumulated_bbox_area"]
        for i in instances
        if i["detector_class_id"] == 2
    }
    legacy_track = (
        max(car_areas, key=lambda track: (car_areas[track], -track))
        if car_areas
        else None
    )
    cards = write_object_cards(instances, originals, vis / "object_cards")
    payload = {
        "schema_version": 1,
        "selection_required": True,
        "source": {
            "image_dir": str(Path(args.image_dir).resolve())
            if args.image_dir
            else None,
            "video_path": str(Path(args.video_path).resolve())
            if args.video_path
            else None,
            "scene_name": args.scene_name,
            "frame_count": len(frames),
            "frame_size": list(frames[0].size),
            "start_idx": args.start_idx,
            "preprocessed_rgb_sha256": hashlib.sha256(
                b"".join(im.tobytes() for im in frames)
            ).hexdigest(),
        },
        "instances": instances,
        "frame_images": frame_images,
        "evidence_images": [str(scene_path.resolve()), *crop_paths],
        "evidence_frame_indices": evidence_indices,
        "object_cards": cards,
        "legacy_selection": {
            "instance_id": mapping[legacy_track] if legacy_track is not None else None,
            "track_id": legacy_track,
            "rule": "largest_accumulated_bbox_area",
        },
    }
    payload["evidence_sha256"] = {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in payload["evidence_images"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(instances), "output": str(args.output)}))


if __name__ == "__main__":
    main()
