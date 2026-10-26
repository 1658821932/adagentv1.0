#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dggt_root", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--scene_names", default="000")
    parser.add_argument("--sequence_length", type=int, default=80)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--bbox_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dynamic_threshold", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.08)
    parser.add_argument("--min_points", type=int, default=3)
    args = parser.parse_args()

    dggt_root = Path(args.dggt_root).resolve()
    sys.path.insert(0, str(dggt_root))
    os.chdir(dggt_root)

    from dggt.models.vggt import VGGT
    from dggt.utils.pose_enc import pose_encoding_to_extri_intri
    from dggt.utils.geometry import unproject_depth_map_to_point_map
    from datasets.dataset import WaymoOpenDataset

    bbox_payload = json.loads(Path(args.bbox_json).read_text(encoding="utf-8"))
    bboxes_orig = {int(k): np.array(v, dtype=np.float32) for k, v in bbox_payload["bboxes"].items()}
    orig_w, orig_h = bbox_payload.get("image_size", [1280, 720])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[manual-instance] device={device}", flush=True)
    dataset = WaymoOpenDataset(
        args.image_dir,
        scene_names=[args.scene_names],
        sequence_length=args.sequence_length,
        start_idx=0,
        mode=2,
        views=1,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("[manual-instance] loading model/checkpoint", flush=True)
    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("[manual-instance] model ready", flush=True)

    frame_ids = {}
    centroids = {"0": []}
    bboxes = {"0": []}
    per_frame_stats = []
    all_dists = []

    with torch.no_grad():
        print("[manual-instance] loading batch", flush=True)
        batch = next(iter(dataloader))
        images = batch["images"].to(device)
        print(f"[manual-instance] images={tuple(images.shape)}", flush=True)
        sky_mask = batch["masks"].to(device).permute(0, 1, 3, 4, 2)
        bg_mask = (sky_mask == 0).any(dim=-1)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            print("[manual-instance] running model forward", flush=True)
            predictions = model(images)
            print("[manual-instance] model forward done", flush=True)
            h, w = images.shape[-2:]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (h, w))
            depth_map = predictions["depth"][0]
            point_map = unproject_depth_map_to_point_map(depth_map, extrinsics[0], intrinsics[0])[None, ...]
            point_map = torch.from_numpy(point_map).to(device).float()
            dy_prob = torch.sigmoid(predictions["dynamic_conf"].squeeze(-1))
            print("[manual-instance] geometry tensors ready", flush=True)

        scale_x = float(w) / float(orig_w)
        scale_y = float(h) / float(orig_h)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        for frame_idx in range(point_map.shape[1]):
            if frame_idx % 10 == 0:
                print(f"[manual-instance] frame {frame_idx}/{point_map.shape[1]}", flush=True)
            dyn_mask = bg_mask[:, frame_idx] & (dy_prob[:, frame_idx] > args.dynamic_threshold)
            dyn_points = point_map[:, frame_idx][dyn_mask].reshape(-1, 3)

            ids = torch.full((len(dyn_points),), -1, dtype=torch.int64, device=device)
            if frame_idx in bboxes_orig and len(dyn_points) > 0:
                x1, y1, x2, y2 = bboxes_orig[frame_idx]
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                in_box_mask = dyn_mask & (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
                selected = point_map[:, frame_idx][in_box_mask].reshape(-1, 3)
                if len(selected) >= args.min_points:
                    centroid = selected.mean(dim=0)
                    bmin = selected.min(dim=0).values
                    bmax = selected.max(dim=0).values
                    dists = torch.linalg.norm(selected - centroid[None, :], dim=1)
                    all_dists.extend(dists.detach().cpu().numpy().tolist())
                    # Fill frame_ids for visualization/debug. Editing will use centroid matching.
                    dyn_flat_idx = torch.nonzero(dyn_mask.reshape(-1), as_tuple=False).squeeze(1)
                    box_flat_idx = torch.nonzero(in_box_mask.reshape(-1), as_tuple=False).squeeze(1)
                    pos = torch.searchsorted(dyn_flat_idx, box_flat_idx)
                    valid = (pos >= 0) & (pos < len(ids)) & (dyn_flat_idx[pos] == box_flat_idx)
                    ids[pos[valid]] = 0
                    centroids["0"].append(centroid.detach().cpu().numpy().tolist())
                    bboxes["0"].append((bmin.detach().cpu().numpy().tolist(), bmax.detach().cpu().numpy().tolist()))
                    per_frame_stats.append({
                        "frame": frame_idx,
                        "dynamic_points": int(len(dyn_points)),
                        "selected_points": int(len(selected)),
                        "radius_p95": float(torch.quantile(dists, 0.95).item()),
                        "radius_max": float(dists.max().item()),
                    })
                else:
                    centroids["0"].append([float("nan")] * 3)
                    bboxes["0"].append(([float("nan")] * 3, [float("nan")] * 3))
                    per_frame_stats.append({
                        "frame": frame_idx,
                        "dynamic_points": int(len(dyn_points)),
                        "selected_points": int(len(selected)),
                    })
            else:
                centroids["0"].append([float("nan")] * 3)
                bboxes["0"].append(([float("nan")] * 3, [float("nan")] * 3))
                per_frame_stats.append({
                    "frame": frame_idx,
                    "dynamic_points": int(len(dyn_points)),
                    "selected_points": 0,
                })

            frame_ids[str(frame_idx)] = ids.detach().cpu().tolist()

    if all_dists:
        d = np.array(all_dists, dtype=np.float32)
        suggested_eps = max(float(np.percentile(d, 95)) / 3.0, args.eps)
    else:
        suggested_eps = args.eps

    payload = {
        "eps": suggested_eps,
        "min_samples": 1,
        "max_match_dist": 2.0,
        "num_instances": 1,
        "frame_ids": frame_ids,
        "centroids": centroids,
        "bboxes": bboxes,
        "manual_bbox_source": str(Path(args.bbox_json).resolve()),
        "manual_stats": per_frame_stats,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved {out_path}")
    print(f"frames: {len(frame_ids)}")
    print(f"suggested eps: {suggested_eps:.6f}  match_dist: {suggested_eps * 3:.6f}")
    print("selected point stats:")
    for item in per_frame_stats:
        if item["frame"] % 10 == 0 or item["frame"] == args.sequence_length - 1:
            print(item)


if __name__ == "__main__":
    main()
