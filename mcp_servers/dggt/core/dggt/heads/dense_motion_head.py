"""Continuous-time dense Gaussian displacement prediction."""

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from dggt.heads.dpt_head import DPTHead


class ContinuousTimeEncoder(nn.Module):
    """Encode continuous source/target times without a discrete time table."""

    def __init__(self, condition_dim: int, num_frequencies: int = 8, hidden_dim: int = None):
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        hidden_dim = hidden_dim or condition_dim
        frequencies = (2.0 ** torch.arange(num_frequencies)) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        encoded_dim = 3 + 3 * 2 * num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    def forward(self, source_times: torch.Tensor, target_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pair conditions and signed deltas with layout [B, S, K, ...]."""
        if source_times.ndim != 2 or target_times.ndim != 2:
            raise ValueError("source_times and target_times must have shape [B, time]")
        if source_times.shape[0] != target_times.shape[0]:
            raise ValueError("source_times and target_times must have the same batch size")

        source = source_times[:, :, None]
        target = target_times[:, None, :]
        signed_delta_t = target - source
        scalars = torch.stack(
            [source.expand_as(signed_delta_t), target.expand_as(signed_delta_t), signed_delta_t], dim=-1
        )
        phases = scalars.unsqueeze(-1) * self.frequencies.to(dtype=scalars.dtype)
        encoded = torch.cat([scalars, phases.sin().flatten(-2), phases.cos().flatten(-2)], dim=-1)
        return self.mlp(encoded), signed_delta_t


class DenseMotionHead(nn.Module):
    """Predict direct world-coordinate displacement for every source-target pair.

    The returned motion is a final displacement, never a velocity. Callers must
    use ``target_xyz = source_xyz + motion`` and must not scale it by time.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        intermediate_layer_idx: Sequence[int] = (4, 11, 17, 23),
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        num_time_frequencies: int = 8,
        pair_chunk_size: int = 1,
    ) -> None:
        super().__init__()
        if len(intermediate_layer_idx) != 4:
            raise ValueError("DenseMotionHead requires exactly four DPT feature layers")

        self.dim_in = dim_in
        self.patch_size = patch_size
        self.pair_chunk_size = pair_chunk_size
        self.intermediate_layer_idx = tuple(intermediate_layer_idx)
        self.time_encoder = ContinuousTimeEncoder(dim_in, num_time_frequencies)
        self.to_film = nn.Linear(dim_in, 2 * dim_in)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

        self.decoder = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            out_channels=list(out_channels),
            intermediate_layer_idx=(0, 1, 2, 3),
            feature_only=True,
        )
        self.motion_projection = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(features, 3, kernel_size=1, stride=1, padding=0),
        )
        nn.init.zeros_(self.motion_projection[-1].weight)
        nn.init.zeros_(self.motion_projection[-1].bias)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        images: torch.Tensor,
        patch_start_idx: int,
        source_indices: torch.Tensor = None,
        target_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return direct motion displacement with public layout [B,S,K,H,W,3]."""
        if images.ndim != 5:
            raise ValueError("images must have shape [B, S, 3, H, W]")
        if not aggregated_tokens_list:
            raise ValueError("aggregated_tokens_list cannot be empty")

        batch_size, source_count, _, height, width = images.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("DenseMotionHead requires H and W divisible by patch_size")
        if source_times.shape != (batch_size, source_count):
            raise ValueError("source_times must have shape [B, S] matching images")
        if target_times.ndim != 2 or target_times.shape[0] != batch_size:
            raise ValueError("target_times must have shape [B, K]")

        target_count = target_times.shape[1]
        if (source_indices is None) != (target_indices is None):
            raise ValueError("source_indices and target_indices must be provided together")
        if source_indices is None:
            pair_source_indices = torch.arange(source_count, device=images.device).repeat_interleave(target_count)
            pair_target_indices = torch.arange(target_count, device=images.device).repeat(source_count)
        else:
            pair_source_indices = source_indices.to(device=images.device, dtype=torch.long)
            pair_target_indices = target_indices.to(device=images.device, dtype=torch.long)
            if pair_source_indices.ndim != 1 or pair_target_indices.shape != pair_source_indices.shape:
                raise ValueError("motion pair indices must be one-dimensional and have equal length")
            if pair_source_indices.numel() == 0:
                raise ValueError("motion pair indices cannot be empty")
            if pair_source_indices.min() < 0 or pair_source_indices.max() >= source_count:
                raise ValueError("source_indices contain an invalid source index")
            if pair_target_indices.min() < 0 or pair_target_indices.max() >= target_count:
                raise ValueError("target_indices contain an invalid target index")
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        patch_count = patch_h * patch_w
        time_condition, signed_delta_t = self.time_encoder(source_times, target_times)
        gamma, beta = self.to_film(time_condition).chunk(2, dim=-1)

        patch_layers = []
        for layer_idx in self.intermediate_layer_idx:
            if layer_idx >= len(aggregated_tokens_list):
                raise ValueError(f"missing DGGT aggregated layer {layer_idx}")
            layer_tokens = aggregated_tokens_list[layer_idx]
            if layer_tokens.shape[:2] != (batch_size, source_count):
                raise ValueError("aggregated token batch/source dimensions do not match images")
            if layer_tokens.shape[-1] != self.dim_in:
                raise ValueError("aggregated token channel dimension does not match DenseMotionHead")
            if layer_tokens.shape[2] - patch_start_idx != patch_count:
                raise ValueError("P - patch_start_idx must equal patch_h * patch_w")

            # Remove camera/register tokens before any spatial DPT reshape.
            patch_tokens = layer_tokens[:, :, patch_start_idx:, :]
            assert patch_tokens.shape[2] == patch_count
            patch_layers.append(patch_tokens)

        pair_count = pair_source_indices.numel()
        motion_chunks = []
        for pair_start in range(0, pair_count, self.pair_chunk_size):
            pair_end = min(pair_start + self.pair_chunk_size, pair_count)
            chunk_source_indices = pair_source_indices[pair_start:pair_end]
            chunk_target_indices = pair_target_indices[pair_start:pair_end]
            chunk_layers = []
            for layer in patch_layers:
                chunk = layer[:, chunk_source_indices]
                chunk_gamma = gamma[:, chunk_source_indices, chunk_target_indices]
                chunk_beta = beta[:, chunk_source_indices, chunk_target_indices]
                chunk_layers.append(
                    chunk * (1.0 + chunk_gamma[:, :, None, :]) + chunk_beta[:, :, None, :]
                )
            pair_images = images[:, chunk_source_indices]
            decoded_features = self.decoder(chunk_layers, pair_images, patch_start_idx=0)
            if decoded_features.shape[:2] != (batch_size, pair_end - pair_start):
                raise RuntimeError("DPT decoder did not preserve the expected pair layout")
            if decoded_features.shape[-2:] != (height, width):
                raise RuntimeError("DPT decoder output resolution differs from source images")
            motion_chunks.append(self.motion_projection(
                decoded_features.reshape(
                    batch_size * (pair_end - pair_start),
                    decoded_features.shape[2], height, width,
                )
            ).reshape(batch_size, pair_end - pair_start, 3, height, width))
        # Restore public [B,S,K,H,W,3], never exchanging source and target dimensions.
        pair_motion = torch.cat(motion_chunks, dim=1)
        motion = pair_motion.new_zeros(batch_size, source_count, target_count, 3, height, width)
        motion[:, pair_source_indices, pair_target_indices, :, :, :] = pair_motion
        motion = motion.permute(0, 1, 2, 4, 5, 3).contiguous()
        assert motion.shape == (batch_size, source_count, target_count, height, width, 3)

        # Identity-pair constraint only; this is not time-based output scaling.
        identity_pairs = signed_delta_t.eq(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return motion.masked_fill(identity_pairs, 0.0)
