# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from dggt.models.aggregator import Aggregator
from dggt.heads.camera_head import CameraHead
from dggt.heads.dpt_head import DPTHead, GaussianHead
from dggt.heads.dense_motion_head import DenseMotionHead
from dggt.heads.track_head import TrackHead
from dggt.heads.gs_head import GaussianDecoder
from dggt.models.sky import SkyGaussian
from dggt.models.fusion import PointNetGSFusion
#from dggt.splatformer.feature_predictor import FeaturePredictor


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, semantic_num = 10):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")# ,down_ratio=2)
        #self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")# ,down_ratio=2)
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="sigmoid")

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        #GS attributes
        self.gs_head = GaussianHead(dim_in= 3 * embed_dim, output_dim=3 + 1 + 3 + 4 + 1 , activation="sigmoid")# ,down_ratio=2)#RGB
        self.instance_head = DPTHead(dim_in= embed_dim, output_dim = 1 + 1, activation="linear") # dynamic logit + confidence
        self.semantic_head = DPTHead(dim_in= embed_dim, output_dim = semantic_num + 1, activation="linear")# ,down_ratio=2)#RGB
        self.dense_motion_head = DenseMotionHead(dim_in=2 * embed_dim, patch_size=patch_size)
        # Color, opacity, scale, rotation
        self.sky_model = SkyGaussian()
        #self.fusion_model = PointNetGSFusion()
        #self.splatformer = FeaturePredictor()
        #self.point_offset_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log_1")


    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        source_times: torch.Tensor = None,
        target_times: torch.Tensor = None,
        predict_dense_motion: bool = False,
        motion_source_indices: torch.Tensor = None,
        motion_target_indices: torch.Tensor = None,
    ):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, image_tokens_list, dino_token_list, image_feature, patch_start_idx = self.aggregator(images)
        
        predictions = {}

        predictions["image_feature"] = image_feature

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

 
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf


            if self.gs_head is not None:
                gs_map, gs_conf = self.gs_head(image_tokens_list, images, patch_start_idx)
                predictions["gs_map"] = gs_map
                predictions["gs_conf"] = gs_conf

            if self.instance_head is not None:
                dynamic_conf, _ = self.instance_head(dino_token_list, images, patch_start_idx)
                predictions["dynamic_conf"] = dynamic_conf

            if self.semantic_head is not None:
                semantic_logits, _ = self.semantic_head(dino_token_list, images, patch_start_idx)
                predictions["semantic_logits"] = semantic_logits

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # Opt-in only: legacy model(images) calls do not run the new head.
            if predict_dense_motion:
                if source_times is None or target_times is None:
                    raise ValueError("source_times and target_times are required for dense motion")
                source_times = source_times.to(device=images.device, dtype=images.dtype)
                target_times = target_times.to(device=images.device, dtype=images.dtype)
                predictions["dense_motion"] = self.dense_motion_head(
                    aggregated_tokens_list,
                    source_times,
                    target_times,
                    images,
                    patch_start_idx,
                    motion_source_indices,
                    motion_target_indices,
                )


        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
