# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F

from .triton_ops import TRITON_AVAILABLE, fused_qkv_qknorm_rope_bf16, linear_bf16_cached

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        use_bf16_cache = os.environ.get("VGGT_OPT_AGG_BF16_CACHE") == "1"
        qkv_raw = linear_bf16_cached(self.qkv, x) if use_bf16_cache else self.qkv(x)

        use_triton_rope = (
            os.environ.get("VGGT_OPT_TRITON_ROPE") == "1"
            and TRITON_AVAILABLE
            and not x.requires_grad
            and not self.qkv.weight.requires_grad
            and x.is_cuda
            and self.head_dim == 64
            and self.fused_attn
            and qkv_raw.dtype == torch.bfloat16
            and (isinstance(self.q_norm, nn.LayerNorm) or self.rope is not None)
            and (self.rope is None or hasattr(self.rope, "get_triton_cos_sin_16"))
        )
        if use_triton_rope:
            cs16 = None
            frame_p = N
            if self.rope is not None:
                res = self.rope.get_triton_cos_sin_16(self.head_dim // 2, qkv_raw.device)
                if res is None:
                    use_triton_rope = False
                else:
                    cs16, frame_p = res

        if use_triton_rope:
            q, k, v = fused_qkv_qknorm_rope_bf16(
                qkv_raw.contiguous(), self.num_heads, self.q_norm, self.k_norm, cs16, frame_p
            )
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            qkv = qkv_raw.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = linear_bf16_cached(self.proj, x) if use_bf16_cache else self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
