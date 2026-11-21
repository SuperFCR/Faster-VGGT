# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._active_rope_cache: Optional[Dict[Tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        self._active_triton_cs16: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._active_pos_1: Optional[torch.Tensor] = None
        self._active_max_pos: Optional[int] = None
        self._active_seq_len_s: Optional[int] = None

    def set_frame_context(self, pos_1: torch.Tensor, max_pos: int, seq_len_s: int):
        """Set per-forward 1-frame position metadata so all 48 blocks share cos/sin tables without GPU->CPU sync."""
        self._active_pos_1 = pos_1
        self._active_max_pos = max_pos
        self._active_seq_len_s = seq_len_s
        self._active_rope_cache = {}
        self._active_triton_cs16 = None

    def clear_frame_context(self):
        self._active_pos_1 = None
        self._active_max_pos = None
        self._active_seq_len_s = None
        self._active_rope_cache = None
        self._active_triton_cs16 = None

    def get_triton_cos_sin_16(
        self, feature_dim: int, device: torch.device
    ) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], int]]:
        """Return compact 1-frame (P, 16) FP32 (cy, sy, cx, sx) tables for Triton fused QK-Norm+RoPE."""
        if self._active_pos_1 is None or feature_dim != 32:
            return None
        if self._active_triton_cs16 is None:
            cos_comp, sin_comp = self._compute_frequency_components(feature_dim, self._active_max_pos, device, torch.float32)
            p0 = self._active_pos_1[0]  # (P, 2)
            cy = F.embedding(p0[:, 0], cos_comp)[:, :16].contiguous()
            sy = F.embedding(p0[:, 0], sin_comp)[:, :16].contiguous()
            cx = F.embedding(p0[:, 1], cos_comp)[:, :16].contiguous()
            sx = F.embedding(p0[:, 1], sin_comp)[:, :16].contiguous()
            self._active_triton_cs16 = (cy, sy, cx, sx)
        return self._active_triton_cs16, int(self._active_pos_1.shape[1])

    def _get_cached_cos_sin(
        self, n_tokens: int, feature_dim: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._active_rope_cache is None or self._active_pos_1 is None:
            return None
        key = (n_tokens, feature_dim, device, dtype)
        cached = self._active_rope_cache.get(key)
        if cached is not None:
            return cached
        P = self._active_pos_1.shape[1]
        S = self._active_seq_len_s
        if n_tokens != P and n_tokens != S * P:
            return None
        base_key = (P, feature_dim, device, dtype)
        base = self._active_rope_cache.get(base_key)
        if base is None:
            cos_comp, sin_comp = self._compute_frequency_components(feature_dim, self._active_max_pos, device, dtype)
            cy = F.embedding(self._active_pos_1[..., 0], cos_comp)[:, None, :, :]
            sy = F.embedding(self._active_pos_1[..., 0], sin_comp)[:, None, :, :]
            cx = F.embedding(self._active_pos_1[..., 1], cos_comp)[:, None, :, :]
            sx = F.embedding(self._active_pos_1[..., 1], sin_comp)[:, None, :, :]
            base = (cy, sy, cx, sx)
            self._active_rope_cache[base_key] = base
        if n_tokens == P:
            return base
        cy, sy, cx, sx = base
        glob = (cy.repeat(1, 1, S, 1), sy.repeat(1, 1, S, 1), cx.repeat(1, 1, S, 1), sx.repeat(1, 1, S, 1))
        self._active_rope_cache[key] = glob
        return glob

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        if os.environ.get("VGGT_OPT_ROPE_CACHE") == "1":
            cached = self._get_cached_cos_sin(tokens.size(2), feature_dim, tokens.device, tokens.dtype)
            if cached is not None:
                cy, sy, cx, sx = cached
                vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
                vertical_features = (vertical_features * cy) + (self._rotate_features(vertical_features) * sy)
                horizontal_features = (horizontal_features * cx) + (self._rotate_features(horizontal_features) * sx)
                return torch.cat((vertical_features, horizontal_features), dim=-1)

        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)
