#!/usr/bin/env python3
"""Run DGGT inference once with replace placement forced to tracker centroids."""
import runpy
import os
import sys

import torch

DGGt_ROOT = "/nas/users/guoluosong/ghc/dggt"


def patch_replace_to_tracker_centroid():
    from omegaconf import OmegaConf
    from utils.edit_gaussian import DynamicGS, DynamicGaussianEditor, _align_gs_to_centroid

    def _apply_replace_tracker_center(self, frame_idx: int, dyn_gs: DynamicGS, cfg) -> DynamicGS:
        assert len(cfg.target_id) == len(cfg.replace_obj), (
            "replace: target_id and replace_obj must have the same length"
        )

        for target_id, replace_obj in zip(cfg.target_id, cfg.replace_obj):
            mask = self._instance_mask(dyn_gs, target_id)
            if not mask.any():
                continue

            orig_centroid = dyn_gs.points[mask].mean(dim=0)
            if self.tracker is not None:
                tracker_centroid = self.tracker.get_instance_centroid(int(target_id), frame_idx)
                if tracker_centroid is not None and not torch.isnan(tracker_centroid).any():
                    orig_centroid = tracker_centroid.to(
                        device=dyn_gs.device,
                        dtype=dyn_gs.points.dtype,
                    )

            orig_pts = dyn_gs.points[mask]
            orig_size = (orig_pts.max(dim=0).values - orig_pts.min(dim=0).values).norm().item()

            keep = ~mask
            dyn_gs = dyn_gs.mask_select(keep)

            if isinstance(replace_obj, str):
                replacement = self._load_ply(replace_obj, dyn_gs.device)
                repl_size = (
                    replacement.points.max(dim=0).values
                    - replacement.points.min(dim=0).values
                ).norm().item()
                scale_factor = (orig_size / repl_size) if repl_size > 1e-6 else 1.0
                scale_factor *= float(getattr(cfg, "scale_multiplier", 1.0))
                rotation_euler = (
                    list(cfg.rotation)
                    if hasattr(cfg, "rotation") and cfg.rotation is not None
                    else None
                )
                replacement = _align_gs_to_centroid(
                    replacement,
                    orig_centroid,
                    scale_factor,
                    rotation_euler,
                )
            elif isinstance(replace_obj, int):
                src_mask = self._instance_mask(dyn_gs, replace_obj)
                if not src_mask.any():
                    continue
                replacement = dyn_gs.mask_select(src_mask)
                replacement = _align_gs_to_centroid(replacement, orig_centroid)
            else:
                raise ValueError(
                    f"replace_obj must be str (PLY path) or int (instance id), got {type(replace_obj)}"
                )

            new_ids = torch.full(
                (len(replacement),),
                target_id,
                dtype=torch.int64,
                device=dyn_gs.device,
            )
            replacement = DynamicGS(
                points=replacement.points,
                rgbs=replacement.rgbs,
                opacities=replacement.opacities,
                scales=replacement.scales,
                rotations=replacement.rotations,
                inst_ids=new_ids,
            )
            dyn_gs = dyn_gs.concat(replacement)

        return dyn_gs

    DynamicGaussianEditor._apply_replace = _apply_replace_tracker_center


def main():
    os.chdir(DGGt_ROOT)
    if DGGt_ROOT not in sys.path:
        sys.path.insert(0, DGGt_ROOT)
    patch_replace_to_tracker_centroid()
    runpy.run_path("/nas/users/guoluosong/ghc/dggt/inference.py", run_name="__main__")


if __name__ == "__main__":
    main()
