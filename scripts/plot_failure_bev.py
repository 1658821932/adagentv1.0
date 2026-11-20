#!/usr/bin/env python3
"""Plot full EGO and target bodies for one archived closed-loop failure.

Example:
    python scripts/plot_failure_bev.py \
        --archive /path/to/navtest_failures_R_complete \
        --token 4abb45d9ee1a5c5c \
        --out /tmp/bev.png

The output uses the official map and recorded EGO path. Every vehicle whose body
comes within the configurable radius of EGO during the selected frames is drawn
with its trajectory, complete oriented body and heading. The viewport is sized
from the complete bodies, so vehicle footprints are never cut off.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import structure_facts as sf  # noqa: E402


def frame_range(value, violation, frame_count):
    """Parse START:END, defaulting to frame 0 through the frame after violation."""
    if value:
        try:
            start_text, end_text = value.split(":", 1)
            start, end = int(start_text), int(end_text)
        except (AttributeError, TypeError, ValueError):
            raise argparse.ArgumentTypeError("frame range must be START:END") from None
    else:
        start, end = 0, min(violation + 1, frame_count - 1)
    if not 0 <= start <= end < frame_count:
        raise argparse.ArgumentTypeError(
            f"frame range {start}:{end} is outside recorded frames 0:{frame_count - 1}")
    return start, end


def projected_converter(frames, intersections):
    """Choose a metric CRS and convert UTM ego/actor coordinates into it."""
    extent = intersections.to_crs("EPSG:4326").total_bounds
    pose = np.asarray(frames[0]["ego2global"], dtype=float)
    lonlat = sf.utm_to_lonlat(float(pose[0, 3]), float(pose[1, 3]), extent)
    metric = sf.metric_crs(intersections.crs, *lonlat)
    transformer = Transformer.from_crs("EPSG:4326", metric, always_xy=True)

    def transform(x, y):
        return transformer.transform(*sf.utm_to_lonlat(float(x), float(y), extent))

    def heading(x, y, yaw):
        x0, y0 = transform(x, y)
        x1, y1 = transform(x + math.cos(yaw), y + math.sin(yaw))
        return x0, y0, math.atan2(y1 - y0, x1 - x0)

    return metric, transform, heading


def transformed_body(polygon, transform):
    points = polygon.exterior.coords if hasattr(polygon, "exterior") else polygon.points
    return np.asarray([transform(x, y) for x, y in points], dtype=float)


def body_center(points):
    return points[:-1].mean(axis=0)


def source_body_center(body):
    points = body.points if hasattr(body, "points") else body.exterior.coords
    return np.asarray(points, dtype=float)[:-1].mean(axis=0)


def draw_body(axis, points, heading, color, label, *, alpha=0.55, zorder=8,
              show_label=True):
    axis.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="black",
                           linewidth=0.75, alpha=alpha, zorder=zorder))
    center = body_center(points)
    length = 4.2
    axis.add_patch(FancyArrowPatch(
        tuple(center), (center[0] + length * math.cos(heading),
                        center[1] + length * math.sin(heading)),
        arrowstyle="-|>", mutation_scale=9, linewidth=0.75, color="black",
        zorder=zorder + 1))
    if show_label:
        axis.text(center[0], center[1] + 1.1, label, fontsize=6, color=color,
                  ha="center", va="bottom", zorder=zorder + 2)


def map_layers(map_location):
    path = sf.map_file(map_location)
    return {
        name: sf.map_layer(gpd, path, name)
        for name in ("lanes_polygons", "baseline_paths", "road_segments", "intersections")
    }


def plot_failure_bev(archive, token, output, *, requested_range=None,
                     nearby_radius_m=30.0, highlight_track=None,
                     margin_m=8.0, dpi=150, title=None):
    archive = Path(archive).expanduser().resolve()
    frames, full_token = sf.scene_pickle(str(archive), token)
    outcome, official_token, violation = sf.outcome(str(archive), full_token)
    tracks = sf.actor_sequences(frames)
    start, end = frame_range(requested_range, violation, len(frames))
    indices = list(range(start, end + 1))

    layers = map_layers(frames[0]["map_location"])
    metric, transform, heading = projected_converter(frames, layers["intersections"])
    layers = {name: frame.to_crs(metric) for name, frame in layers.items()}
    ego_points = []
    body_geometry = []

    figure, axis = plt.subplots(figsize=(13, 9.5))
    layers["lanes_polygons"].plot(ax=axis, facecolor="#edf1f5", edgecolor="#9ca8b6",
                                  linewidth=0.55, zorder=0)
    layers["road_segments"].plot(ax=axis, facecolor="none", edgecolor="#596573",
                                  linewidth=0.8, zorder=1)
    layers["baseline_paths"].plot(ax=axis, color="#d5a21c", linewidth=0.55,
                                   alpha=0.75, zorder=2)
    layers["intersections"].boundary.plot(ax=axis, color="black", linewidth=1.6, zorder=3)

    nearby = []
    for track, observations in sorted(tracks.items()):
        rows = {item["index"]: item for item in observations if start <= item["index"] <= end}
        if not rows:
            continue
        within_radius = False
        for index in rows:
            ego = sf.ego_body(frames[index], index)
            actor = sf.actor_body(rows[index], frames[index], index)
            distance = ego.distance(actor)
            if math.isfinite(distance) and distance <= nearby_radius_m:
                within_radius = True
                break
        if within_radius:
            nearby.append((track, rows))

    for index in indices:
        ego_utm = sf.ego_body(frames[index], index)
        ego_shape = transformed_body(ego_utm, transform)
        center = source_body_center(ego_utm)
        x, y, yaw = heading(center[0], center[1], sf.ego_heading(frames[index], index))
        draw_body(axis, ego_shape, yaw, "#00897b", f"E{index}")
        ego_points.append((x, y))
        body_geometry.extend(ego_shape)

    ego_points = np.asarray(ego_points)
    axis.plot(ego_points[:, 0], ego_points[:, 1], color="#1976d2", linewidth=2.5,
              label="EGO trajectory", zorder=6)
    palette = plt.get_cmap("tab20")
    handles = [Line2D([0], [0], color="#1976d2", linewidth=2.5, label="EGO")]
    actor_records = []
    for number, (track, rows) in enumerate(nearby, start=1):
        color = palette((number - 1) % 20)
        points = []
        for index in sorted(rows):
            item = rows[index]
            actor_utm = sf.actor_body(item, frames[index], index)
            actor_shape = transformed_body(actor_utm, transform)
            center = source_body_center(actor_utm)
            x, y, yaw = heading(center[0], center[1], item["world_yaw"])
            draw_body(axis, actor_shape, yaw, color, f"V{number}", alpha=0.38,
                      zorder=7, show_label=index == min(rows))
            points.append((x, y))
            body_geometry.extend(actor_shape)
        points = np.asarray(points)
        axis.plot(points[:, 0], points[:, 1], color=color,
                  linewidth=2.8 if track == highlight_track else 1.8,
                  alpha=0.95 if track == highlight_track else 0.8, zorder=6)
        axis.text(points[0, 0], points[0, 1], f"V{number}", fontsize=7,
                  color=color, fontweight="bold", zorder=12)
        handles.append(Line2D([0], [0], color=color, linewidth=1.8,
                              label=f"V{number} {track[:8]}"))
        actor_records.append({"vehicle": f"V{number}", "track": track,
                              "frames_observed": sorted(rows)})

    geometry = np.asarray(body_geometry)
    low = geometry.min(axis=0) - margin_m
    high = geometry.max(axis=0) + margin_m
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.set_xlabel("map x (m)")
    axis.set_ylabel("map y (m)")
    axis.legend(handles=handles, loc="upper right", fontsize=6.5,
                framealpha=0.9, ncol=1 if len(handles) <= 8 else 2)
    axis.set_title(title or (
        f"BEV OVERVIEW frame {start:02d}-{end:02d}: complete EGO"
        + f" and {len(nearby)} nearby vehicles + headings"))
    figure.tight_layout()

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "archive": str(archive),
        "token": full_token,
        "official_token": official_token,
        "outcome": outcome,
        "violation_step": violation,
        "frame_range": [start, end],
        "nearby_radius_m": nearby_radius_m,
        "nearby_vehicles": actor_records,
        "image": str(output),
    }, ensure_ascii=False, indent=2))
    return {"image": str(output), "metadata": str(sidecar),
            "nearby_vehicles": actor_records,
            "violation_step": violation, "frame_range": [start, end], "outcome": outcome}


def score_zero_tokens(archive):
    path = Path(archive) / "scene_index.csv"
    if not path.is_file():
        raise ValueError(f"missing scene_index.csv: {path}")
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                score = float(row.get("score") or "nan")
            except ValueError:
                raise ValueError(f"invalid score for token {row.get('token')!r}") from None
            if math.isfinite(score) and score == 0.0:
                token = (row.get("token") or "").strip()
                if not token:
                    raise ValueError("score-zero scene has an empty token")
                rows.append(token)
    if not rows:
        raise ValueError(f"no score-zero scenes in {path}")
    return sorted(set(rows))


def _batch_job(arguments):
    token, archive, output, nearby_radius_m, highlight_track, margin_m, dpi = arguments
    if Path(output).is_file() and Path(output).with_suffix(".json").is_file():
        return {"token": token, "status": "skipped_existing", "image": output}
    try:
        result = plot_failure_bev(
            archive, token, output, nearby_radius_m=nearby_radius_m,
            highlight_track=highlight_track, margin_m=margin_m, dpi=dpi)
        return {"token": token, "status": "success", **result}
    except Exception as error:
        return {"token": token, "status": "error", "error_type": type(error).__name__,
                "error": str(error)}


def plot_all_score_zero(archive, output_dir, *, workers=4, nearby_radius_m=30.0,
                        highlight_track=None, margin_m=8.0, dpi=150):
    archive = Path(archive).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = score_zero_tokens(archive)
    jobs = [(token, str(archive), str(output_dir / f"{token}.png"), nearby_radius_m,
             highlight_track, margin_m, dpi) for token in tokens]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_batch_job, job): job[0] for job in jobs}
        for future in as_completed(pending):
            records.append(future.result())
    records.sort(key=lambda row: row["token"])
    success = [row for row in records if row["status"] in ("success", "skipped_existing")]
    failed = [row for row in records if row["status"] not in ("success", "skipped_existing")]
    manifest = {"archive": str(archive), "expected": len(tokens), "success": len(success),
                "failed": len(failed), "workers": workers, "records": records}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"output_dir": str(output_dir), "manifest": str(manifest_path),
            "expected": len(tokens), "success": len(success), "failed": len(failed)}


def parser():
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--archive", required=True, type=Path,
                        help="Closed-loop run directory containing scene_index.csv")
    result.add_argument("--token", help="Full scene token or unique suffix")
    result.add_argument("--out", type=Path, help="Output PNG path for one scene")
    result.add_argument("--batch-score-zero", action="store_true",
                        help="Render every score-zero scene in scene_index.csv")
    result.add_argument("--out-dir", type=Path, help="Output directory in batch mode")
    result.add_argument("--workers", type=int, default=4, help="Batch worker count (default: 4)")
    result.add_argument("--frame-range", help="Inclusive START:END; default is 0 through violation + 1")
    result.add_argument("--nearby-radius-m", type=float, default=30.0,
                        help="Include vehicles whose bodies come this close to EGO")
    result.add_argument("--highlight-track", help="Draw one selected vehicle trajectory more prominently")
    result.add_argument("--margin-m", type=float, default=8.0, help="Margin around complete body extents")
    result.add_argument("--dpi", type=int, default=150)
    result.add_argument("--title")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.margin_m < 0:
        raise SystemExit("margin-m must be nonnegative")
    if args.nearby_radius_m < 0:
        raise SystemExit("nearby-radius-m must be nonnegative")
    if args.batch_score_zero:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --batch-score-zero")
        if args.workers < 1:
            raise SystemExit("workers must be positive")
        try:
            result = plot_all_score_zero(
                args.archive, args.out_dir, workers=args.workers,
                nearby_radius_m=args.nearby_radius_m,
                highlight_track=args.highlight_track,
                margin_m=args.margin_m, dpi=args.dpi)
        except (ValueError, OSError, sf.FactsError) as error:
            raise SystemExit(str(error)) from None
        print(json.dumps(result, ensure_ascii=False))
        return
    if not args.token or not args.out:
        raise SystemExit("--token and --out are required outside batch mode")
    try:
        result = plot_failure_bev(
            args.archive, args.token, args.out, requested_range=args.frame_range,
            nearby_radius_m=args.nearby_radius_m, highlight_track=args.highlight_track,
            margin_m=args.margin_m, dpi=args.dpi, title=args.title)
    except (ValueError, OSError, sf.FactsError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
