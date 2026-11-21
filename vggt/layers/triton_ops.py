"""Custom Triton kernels and frozen-weight BF16 caching for VGGT Aggregator & VisionTransformer.

All fast paths are guarded by explicit environment switches (OFF by default):
  - VGGT_OPT_AGG_BF16_CACHE=1 : Cache BF16 copies of frozen (requires_grad=False) Linear weights/biases
                                 keyed by tensor _version (zero state_dict/parameter changes, N0 bitwise).
  - VGGT_OPT_TRITON_ROPE=1    : Fused QKV unpack + QK-Norm + 2D RoPE -> contiguous (B, H, N, 64) BF16
                                 (replaces ~28 unfused kernels per block with 1 Triton kernel, N1).
  - VGGT_OPT_TRITON_NORM=1    : Fused LayerNorm->BF16 and LayerScale+Residual+LayerNorm->BF16 Triton kernels
                                 preserving exact FP32 residual stream while avoiding FP32->BF16 casts (N1).
"""
import os
import random
import struct
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Protect Python global RNG from Triton's file cache temp-name generation (ResearchBoost lessons.md)
def _preserve_py_random(fn, *args, **kwargs):
    st = random.getstate()
    try:
        return fn(*args, **kwargs)
    finally:
        random.setstate(st)


# ---------------------------------------------------------------------------
# 1. Frozen BF16 Linear Weight Cache (N0 bitwise identical to autocast bf16)
# ---------------------------------------------------------------------------
_BF16_LINEAR_CACHE = {}


def linear_bf16_cached(mod: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Fast path for frozen nn.Linear under bf16 autocast: avoids casting ~908M FP32 weights every microbatch."""
    w = mod.weight
    b = mod.bias
    if (
        not w.requires_grad
        and (b is None or not b.requires_grad)
        and torch.is_autocast_enabled()
        and torch.get_autocast_gpu_dtype() == torch.bfloat16
        and w.is_cuda
        and w.dtype == torch.float32
    ):
        key = id(mod)
        wv = w._version
        bv = b._version if b is not None else -1
        entry = _BF16_LINEAR_CACHE.get(key)
        if entry is None or entry[0] != wv or entry[1] != bv or entry[2].device != w.device:
            w_bf16 = w.detach().to(torch.bfloat16)
            b_bf16 = b.detach().to(torch.bfloat16) if b is not None else None
            entry = (wv, bv, w_bf16, b_bf16)
            _BF16_LINEAR_CACHE[key] = entry
        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        return F.linear(x_bf16, entry[2], entry[3])
    return mod(x)


def conv2d_bf16_cached(mod: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Fast path for frozen nn.Conv2d under bf16 autocast."""
    w = mod.weight
    b = mod.bias
    if (
        not w.requires_grad
        and (b is None or not b.requires_grad)
        and torch.is_autocast_enabled()
        and torch.get_autocast_gpu_dtype() == torch.bfloat16
        and w.is_cuda
        and w.dtype == torch.float32
    ):
        key = id(mod)
        wv = w._version
        bv = b._version if b is not None else -1
        entry = _BF16_LINEAR_CACHE.get(key)
        if entry is None or entry[0] != wv or entry[1] != bv or entry[2].device != w.device:
            w_bf16 = w.detach().to(torch.bfloat16)
            b_bf16 = b.detach().to(torch.bfloat16) if b is not None else None
            entry = (wv, bv, w_bf16, b_bf16)
            _BF16_LINEAR_CACHE[key] = entry
        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        return F.conv2d(x_bf16, entry[2], entry[3], mod.stride, mod.padding, mod.dilation, mod.groups)
    return mod(x)


# ---------------------------------------------------------------------------
# 2. Triton Fused QKV Unpack + QK-Norm + 2D RoPE (exact FP32 internal math)
# ---------------------------------------------------------------------------
if TRITON_AVAILABLE:

    @triton.jit(
        do_not_specialize=[
            "N", "P",
            "stride_qkv_b", "stride_out_b",
            "eps",
        ]
    )
    def _fused_qkv_qknorm_rope_kernel(
        QKV_ptr,
        QW_ptr, QB_ptr,
        KW_ptr, KB_ptr,
        CY_ptr, SY_ptr,
        CX_ptr, SX_ptr,
        Q_out_ptr, K_out_ptr,
        N, P, C,
        stride_qkv_b, stride_qkv_n,
        stride_out_b, stride_out_h, stride_out_n,
        eps,
        H: tl.constexpr,
        HAS_QKNORM: tl.constexpr,
        HAS_ROPE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh = tl.program_id(0)
        n_blk = tl.program_id(1)
        b = bh // H
        h = bh % H

        offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Offsets within 64-dim head split into 4 x 16 quadrants: [v1(0..15), v2(16..31), h1(32..47), h2(48..63)]
        offs_16 = tl.arange(0, 16)
        base_qkv = b * stride_qkv_b + offs_n[:, None] * stride_qkv_n + h * 64
        base_out = b * stride_out_b + h * stride_out_h + offs_n[:, None] * stride_out_n

        # --- Load Q (4 x 16) ---
        q0 = tl.load(QKV_ptr + base_qkv + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        q1 = tl.load(QKV_ptr + base_qkv + 16 + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        q2 = tl.load(QKV_ptr + base_qkv + 32 + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        q3 = tl.load(QKV_ptr + base_qkv + 48 + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)

        # --- Load K (4 x 16) ---
        k_base = base_qkv + C
        k0 = tl.load(QKV_ptr + k_base + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        k1 = tl.load(QKV_ptr + k_base + 16 + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(QKV_ptr + k_base + 32 + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
        k3 = tl.load(QKV_ptr + k_base + 48 + offs_16[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)

        if HAS_QKNORM:
            # Exact FP32 LayerNorm over 64 elements for Q
            q_mean = (tl.sum(q0, axis=1) + tl.sum(q1, axis=1) + tl.sum(q2, axis=1) + tl.sum(q3, axis=1)) * (1.0 / 64.0)
            dq0 = q0 - q_mean[:, None]
            dq1 = q1 - q_mean[:, None]
            dq2 = q2 - q_mean[:, None]
            dq3 = q3 - q_mean[:, None]
            q_var = (
                tl.sum(dq0 * dq0, axis=1)
                + tl.sum(dq1 * dq1, axis=1)
                + tl.sum(dq2 * dq2, axis=1)
                + tl.sum(dq3 * dq3, axis=1)
            ) * (1.0 / 64.0)
            q_rstd = tl.rsqrt(q_var + eps)[:, None]
            qw0 = tl.load(QW_ptr + offs_16).to(tl.float32)[None, :]
            qw1 = tl.load(QW_ptr + 16 + offs_16).to(tl.float32)[None, :]
            qw2 = tl.load(QW_ptr + 32 + offs_16).to(tl.float32)[None, :]
            qw3 = tl.load(QW_ptr + 48 + offs_16).to(tl.float32)[None, :]
            qb0 = tl.load(QB_ptr + offs_16).to(tl.float32)[None, :]
            qb1 = tl.load(QB_ptr + 16 + offs_16).to(tl.float32)[None, :]
            qb2 = tl.load(QB_ptr + 32 + offs_16).to(tl.float32)[None, :]
            qb3 = tl.load(QB_ptr + 48 + offs_16).to(tl.float32)[None, :]
            q0 = dq0 * q_rstd * qw0 + qb0
            q1 = dq1 * q_rstd * qw1 + qb1
            q2 = dq2 * q_rstd * qw2 + qb2
            q3 = dq3 * q_rstd * qw3 + qb3

            # Exact FP32 LayerNorm over 64 elements for K
            k_mean = (tl.sum(k0, axis=1) + tl.sum(k1, axis=1) + tl.sum(k2, axis=1) + tl.sum(k3, axis=1)) * (1.0 / 64.0)
            dk0 = k0 - k_mean[:, None]
            dk1 = k1 - k_mean[:, None]
            dk2 = k2 - k_mean[:, None]
            dk3 = k3 - k_mean[:, None]
            k_var = (
                tl.sum(dk0 * dk0, axis=1)
                + tl.sum(dk1 * dk1, axis=1)
                + tl.sum(dk2 * dk2, axis=1)
                + tl.sum(dk3 * dk3, axis=1)
            ) * (1.0 / 64.0)
            k_rstd = tl.rsqrt(k_var + eps)[:, None]
            kw0 = tl.load(KW_ptr + offs_16).to(tl.float32)[None, :]
            kw1 = tl.load(KW_ptr + 16 + offs_16).to(tl.float32)[None, :]
            kw2 = tl.load(KW_ptr + 32 + offs_16).to(tl.float32)[None, :]
            kw3 = tl.load(KW_ptr + 48 + offs_16).to(tl.float32)[None, :]
            kb0 = tl.load(KB_ptr + offs_16).to(tl.float32)[None, :]
            kb1 = tl.load(KB_ptr + 16 + offs_16).to(tl.float32)[None, :]
            kb2 = tl.load(KB_ptr + 32 + offs_16).to(tl.float32)[None, :]
            kb3 = tl.load(KB_ptr + 48 + offs_16).to(tl.float32)[None, :]
            k0 = dk0 * k_rstd * kw0 + kb0
            k1 = dk1 * k_rstd * kw1 + kb1
            k2 = dk2 * k_rstd * kw2 + kb2
            k3 = dk3 * k_rstd * kw3 + kb3

        if HAS_ROPE:
            p_idx = offs_n % P
            cs_off = p_idx[:, None] * 16 + offs_16[None, :]
            cy = tl.load(CY_ptr + cs_off, mask=mask_n[:, None], other=1.0).to(tl.float32)
            sy = tl.load(SY_ptr + cs_off, mask=mask_n[:, None], other=0.0).to(tl.float32)
            cx = tl.load(CX_ptr + cs_off, mask=mask_n[:, None], other=1.0).to(tl.float32)
            sx = tl.load(SX_ptr + cs_off, mask=mask_n[:, None], other=0.0).to(tl.float32)

            nq0 = q0 * cy - q1 * sy
            nq1 = q1 * cy + q0 * sy
            nq2 = q2 * cx - q3 * sx
            nq3 = q3 * cx + q2 * sx
            q0, q1, q2, q3 = nq0, nq1, nq2, nq3

            nk0 = k0 * cy - k1 * sy
            nk1 = k1 * cy + k0 * sy
            nk2 = k2 * cx - k3 * sx
            nk3 = k3 * cx + k2 * sx
            k0, k1, k2, k3 = nk0, nk1, nk2, nk3

        # Store Q, K in BF16 (direct input to FlashAttention)
        tl.store(Q_out_ptr + base_out + offs_16[None, :], q0.to(tl.bfloat16), mask=mask_n[:, None])
        tl.store(Q_out_ptr + base_out + 16 + offs_16[None, :], q1.to(tl.bfloat16), mask=mask_n[:, None])
        tl.store(Q_out_ptr + base_out + 32 + offs_16[None, :], q2.to(tl.bfloat16), mask=mask_n[:, None])
        tl.store(Q_out_ptr + base_out + 48 + offs_16[None, :], q3.to(tl.bfloat16), mask=mask_n[:, None])

        tl.store(K_out_ptr + base_out + offs_16[None, :], k0.to(tl.bfloat16), mask=mask_n[:, None])
        tl.store(K_out_ptr + base_out + 16 + offs_16[None, :], k1.to(tl.bfloat16), mask=mask_n[:, None])
        tl.store(K_out_ptr + base_out + 32 + offs_16[None, :], k2.to(tl.bfloat16), mask=mask_n[:, None])
        tl.store(K_out_ptr + base_out + 48 + offs_16[None, :], k3.to(tl.bfloat16), mask=mask_n[:, None])

    @triton.jit(do_not_specialize=["stride_x", "stride_y", "eps"])
    def _layernorm_to_bf16_kernel(
        X_ptr, W_ptr, B_ptr, Y_ptr,
        stride_x, stride_y,
        eps,
        C: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, C)
        x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) * (1.0 / C)
        dx = x - mean
        var = tl.sum(dx * dx, axis=0) * (1.0 / C)
        rstd = tl.rsqrt(var + eps)
        w = tl.load(W_ptr + offs).to(tl.float32)
        b = tl.load(B_ptr + offs).to(tl.float32)
        y = dx * rstd * w + b
        tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16))

    @triton.jit(do_not_specialize=["stride_x", "stride_u", "stride_xo", "stride_yo", "eps"])
    def _ls_add_layernorm_bf16_kernel(
        X_ptr, U_ptr, G_ptr, W_ptr, B_ptr,
        X_out_ptr, Y_out_ptr,
        stride_x, stride_u, stride_xo, stride_yo,
        eps,
        C: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, C)
        x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)
        u = tl.load(U_ptr + row * stride_u + offs).to(tl.float32)
        g = tl.load(G_ptr + offs).to(tl.float32)
        xn = x + u * g
        tl.store(X_out_ptr + row * stride_xo + offs, xn)
        mean = tl.sum(xn, axis=0) * (1.0 / C)
        dx = xn - mean
        var = tl.sum(dx * dx, axis=0) * (1.0 / C)
        rstd = tl.rsqrt(var + eps)
        w = tl.load(W_ptr + offs).to(tl.float32)
        b = tl.load(B_ptr + offs).to(tl.float32)
        y = dx * rstd * w + b
        tl.store(Y_out_ptr + row * stride_yo + offs, y.to(tl.bfloat16))

    @triton.jit(do_not_specialize=["stride_x", "stride_u", "stride_xo"])
    def _ls_add_kernel(
        X_ptr, U_ptr, G_ptr, X_out_ptr,
        stride_x, stride_u, stride_xo,
        C: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, C)
        x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)
        u = tl.load(U_ptr + row * stride_u + offs).to(tl.float32)
        g = tl.load(G_ptr + offs).to(tl.float32)
        xn = x + u * g
        tl.store(X_out_ptr + row * stride_xo + offs, xn)

    @triton.jit(
        do_not_specialize=["N", "H_in", "W_in", "H_out", "W_out", "sy_bits_i64", "sx_bits_i64"]
    )
    def _bilinear_nhwc_fwd_kernel(
        X_ptr, Y_ptr,
        N, H_in, W_in, H_out, W_out,
        sy_bits_i64, sx_bits_i64,
        C: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        pid = tl.program_id(0)
        hw_out = H_out * W_out
        n = pid // hw_out
        rem = pid % hw_out
        yo = rem // W_out
        xo = rem % W_out

        scale_y = sy_bits_i64.to(tl.float64, bitcast=True)
        scale_x = sx_bits_i64.to(tl.float64, bitcast=True)
        fy = yo.to(tl.float64) * scale_y
        fx = xo.to(tl.float64) * scale_x
        y0 = tl.floor(fy).to(tl.int32)
        x0 = tl.floor(fx).to(tl.int32)
        y1 = tl.minimum(y0 + 1, H_in - 1)
        x1 = tl.minimum(x0 + 1, W_in - 1)
        wy1 = (fy - y0.to(tl.float64)).to(tl.float32)
        wy0 = 1.0 - wy1
        wx1 = (fx - x0.to(tl.float64)).to(tl.float32)
        wx0 = 1.0 - wx1

        offs_c = tl.arange(0, C)
        base_n = n.to(tl.int64) * H_in * W_in * C
        p00 = tl.load(X_ptr + base_n + (y0 * W_in + x0).to(tl.int64) * C + offs_c).to(tl.float32)
        p01 = tl.load(X_ptr + base_n + (y0 * W_in + x1).to(tl.int64) * C + offs_c).to(tl.float32)
        p10 = tl.load(X_ptr + base_n + (y1 * W_in + x0).to(tl.int64) * C + offs_c).to(tl.float32)
        p11 = tl.load(X_ptr + base_n + (y1 * W_in + x1).to(tl.int64) * C + offs_c).to(tl.float32)

        val = (wy0 * wx0) * p00 + (wy0 * wx1) * p01 + (wy1 * wx0) * p10 + (wy1 * wx1) * p11
        out_base = pid.to(tl.int64) * C + offs_c
        if IS_BF16:
            tl.store(Y_ptr + out_base, val.to(tl.bfloat16))
        else:
            tl.store(Y_ptr + out_base, val)

    @triton.jit(
        do_not_specialize=[
            "N", "H_in", "W_in", "H_out", "W_out",
            "sy_bits_i64", "sx_bits_i64", "inv_scale_y", "inv_scale_x",
        ]
    )
    def _bilinear_nhwc_bwd_kernel(
        GO_ptr, GI_ptr,
        N, H_in, W_in, H_out, W_out,
        sy_bits_i64, sx_bits_i64,
        inv_scale_y, inv_scale_x,
        C: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        # Gather-based backward: 1 program per input pixel (n, yi, xi), ZERO atomics, deterministic!
        pid = tl.program_id(0)
        hw_in = H_in * W_in
        n = pid // hw_in
        rem = pid % hw_in
        yi = rem // W_in
        xi = rem % W_in

        scale_y = sy_bits_i64.to(tl.float64, bitcast=True)
        scale_x = sx_bits_i64.to(tl.float64, bitcast=True)
        yo_start = tl.maximum(0, tl.floor((yi.to(tl.float32) - 1.0) * inv_scale_y).to(tl.int32) - 1)
        yo_end = tl.minimum(H_out - 1, tl.ceil((yi.to(tl.float32) + 1.0) * inv_scale_y).to(tl.int32) + 1)
        xo_start = tl.maximum(0, tl.floor((xi.to(tl.float32) - 1.0) * inv_scale_x).to(tl.int32) - 1)
        xo_end = tl.minimum(W_out - 1, tl.ceil((xi.to(tl.float32) + 1.0) * inv_scale_x).to(tl.int32) + 1)

        offs_c = tl.arange(0, C)
        go_base_n = n.to(tl.int64) * H_out * W_out * C
        acc = tl.zeros([C], dtype=tl.float32)

        yo = yo_start
        while yo <= yo_end:
            fy = yo.to(tl.float64) * scale_y
            y0 = tl.floor(fy).to(tl.int32)
            y1 = tl.minimum(y0 + 1, H_in - 1)
            wy1 = (fy - y0.to(tl.float64)).to(tl.float32)
            wy0 = 1.0 - wy1
            wy = tl.where(y0 == yi, wy0, 0.0) + tl.where(y1 == yi, wy1, 0.0)
            if wy != 0.0:
                xo = xo_start
                while xo <= xo_end:
                    fx = xo.to(tl.float64) * scale_x
                    x0 = tl.floor(fx).to(tl.int32)
                    x1 = tl.minimum(x0 + 1, W_in - 1)
                    wx1 = (fx - x0.to(tl.float64)).to(tl.float32)
                    wx0 = 1.0 - wx1
                    wx = tl.where(x0 == xi, wx0, 0.0) + tl.where(x1 == xi, wx1, 0.0)
                    if wx != 0.0:
                        g = tl.load(GO_ptr + go_base_n + (yo * W_out + xo).to(tl.int64) * C + offs_c).to(tl.float32)
                        acc += (wy * wx) * g
                    xo += 1
            yo += 1

        gi_base = pid.to(tl.int64) * C + offs_c
        if IS_BF16:
            tl.store(GI_ptr + gi_base, acc.to(tl.bfloat16))
        else:
            tl.store(GI_ptr + gi_base, acc)


def fused_qkv_qknorm_rope_bf16(
    qkv: torch.Tensor,
    num_heads: int,
    q_norm: nn.Module,
    k_norm: nn.Module,
    cos_sin_16: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    frame_p: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused QKV unpack + QK-Norm + 2D RoPE in 1 Triton kernel, returning contiguous (B, H, N, 64) BF16."""
    B, N, three_c = qkv.shape
    C = three_c // 3
    head_dim = C // num_heads
    assert head_dim == 64 and qkv.is_contiguous()

    has_qknorm = isinstance(q_norm, nn.LayerNorm) and isinstance(k_norm, nn.LayerNorm)
    has_rope = cos_sin_16 is not None

    if os.environ.get("VGGT_OPT_TRITON_ROPE_NH") == "1":
        # Allocate (B, N, H, 64) transposed layout so FlashAttention returns (B, N, H, 64) and
        # x.transpose(1, 2).reshape(B, N, C) is a zero-copy view (eliminates 480 direct_copy_kernels).
        q_out = torch.empty((B, N, num_heads, 64), device=qkv.device, dtype=torch.bfloat16).transpose(1, 2)
        k_out = torch.empty((B, N, num_heads, 64), device=qkv.device, dtype=torch.bfloat16).transpose(1, 2)
    else:
        q_out = torch.empty((B, num_heads, N, 64), device=qkv.device, dtype=torch.bfloat16)
        k_out = torch.empty((B, num_heads, N, 64), device=qkv.device, dtype=torch.bfloat16)
    # Zero-copy BF16 view for V (stride(-1)==1, consumed directly by FlashAttention)
    v_out = qkv.view(B, N, 3, num_heads, 64)[:, :, 2].transpose(1, 2)

    qw = q_norm.weight if has_qknorm else qkv
    qb = q_norm.bias if has_qknorm else qkv
    kw = k_norm.weight if has_qknorm else qkv
    kb = k_norm.bias if has_qknorm else qkv
    eps = q_norm.eps if has_qknorm else 1e-5

    if has_rope:
        cy, sy, cx, sx = cos_sin_16
    else:
        cy = sy = cx = sx = qkv

    BLOCK_N = 32
    grid = (B * num_heads, triton.cdiv(N, BLOCK_N))

    def _launch():
        _fused_qkv_qknorm_rope_kernel[grid](
            qkv,
            qw, qb, kw, kb,
            cy, sy, cx, sx,
            q_out, k_out,
            N, frame_p, C,
            qkv.stride(0), qkv.stride(1),
            q_out.stride(0), q_out.stride(1), q_out.stride(2),
            eps,
            H=num_heads,
            HAS_QKNORM=has_qknorm,
            HAS_ROPE=has_rope,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

    _preserve_py_random(_launch)
    return q_out, k_out, v_out


def fused_layernorm_bf16(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Exact FP32 LayerNorm outputting BF16 directly (for frozen blocks under bf16 autocast)."""
    C = x.shape[-1]
    x_2d = x.reshape(-1, C).contiguous()
    M = x_2d.shape[0]
    y_2d = torch.empty((M, C), device=x.device, dtype=torch.bfloat16)

    def _launch():
        _layernorm_to_bf16_kernel[(M,)](
            x_2d, norm.weight, norm.bias, y_2d,
            x_2d.stride(0), y_2d.stride(0),
            norm.eps,
            C=C,
            num_warps=4,
        )

    _preserve_py_random(_launch)
    return y_2d.view(*x.shape)


def fused_ls_add_layernorm_bf16(
    x: torch.Tensor, update: torch.Tensor, gamma: torch.Tensor, norm: nn.LayerNorm, inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused (x_new = x + update * gamma) in FP32 + LayerNorm(x_new) -> BF16 in 1 Triton kernel."""
    C = x.shape[-1]
    x_2d = x.reshape(-1, C).contiguous()
    u_2d = update.reshape(-1, C).contiguous()
    M = x_2d.shape[0]
    x_out = x_2d if (inplace and x_2d.dtype == torch.float32) else torch.empty((M, C), device=x.device, dtype=torch.float32)
    y_out = torch.empty((M, C), device=x.device, dtype=torch.bfloat16)

    def _launch():
        _ls_add_layernorm_bf16_kernel[(M,)](
            x_2d, u_2d, gamma, norm.weight, norm.bias,
            x_out, y_out,
            x_2d.stride(0), u_2d.stride(0), x_out.stride(0), y_out.stride(0),
            norm.eps,
            C=C,
            num_warps=4,
        )

    _preserve_py_random(_launch)
    return x_out.view(*x.shape), y_out.view(*x.shape)


def fused_ls_add(x: torch.Tensor, update: torch.Tensor, gamma: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Fused (x_new = x + update * gamma) in FP32 in 1 Triton kernel."""
    C = x.shape[-1]
    x_2d = x.reshape(-1, C).contiguous()
    u_2d = update.reshape(-1, C).contiguous()
    M = x_2d.shape[0]
    x_out = x_2d if (inplace and x_2d.dtype == torch.float32) else torch.empty((M, C), device=x.device, dtype=torch.float32)

    def _launch():
        _ls_add_kernel[(M,)](
            x_2d, u_2d, gamma, x_out,
            x_2d.stride(0), u_2d.stride(0), x_out.stride(0),
            C=C,
            num_warps=4,
        )

    _preserve_py_random(_launch)
    return x_out.view(*x.shape)


class _TritonBilinearNHWC(torch.autograd.Function):
    """Atomic-free, deterministic NHWC bilinear upsample (align_corners=True) in BF16 or FP32."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
        N, C, H_in, W_in = x.shape
        x_nhwc = x.contiguous(memory_format=torch.channels_last)
        y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)

        scale_y = float(H_in - 1) / float(H_out - 1) if H_out > 1 else 0.0
        scale_x = float(W_in - 1) / float(W_out - 1) if W_out > 1 else 0.0
        sy_bits = struct.unpack("<q", struct.pack("<d", scale_y))[0]
        sx_bits = struct.unpack("<q", struct.pack("<d", scale_x))[0]
        is_bf16 = x.dtype == torch.bfloat16

        grid = (N * H_out * W_out,)
        n_warps = 2

        def _launch():
            _bilinear_nhwc_fwd_kernel[grid](
                x_nhwc, y,
                N, H_in, W_in, H_out, W_out,
                sy_bits, sx_bits,
                C=C,
                IS_BF16=is_bf16,
                num_warps=n_warps,
            )

        _preserve_py_random(_launch)
        ctx.meta = (N, C, H_in, W_in, H_out, W_out, scale_y, scale_x, sy_bits, sx_bits, is_bf16, n_warps)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        N, C, H_in, W_in, H_out, W_out, scale_y, scale_x, sy_bits, sx_bits, is_bf16, n_warps = ctx.meta
        go_nhwc = grad_out.contiguous(memory_format=torch.channels_last)
        if go_nhwc.dtype != (torch.bfloat16 if is_bf16 else torch.float32):
            go_nhwc = go_nhwc.to(torch.bfloat16 if is_bf16 else torch.float32)
        gi = torch.empty(
            (N, C, H_in, W_in),
            device=grad_out.device,
            dtype=torch.bfloat16 if is_bf16 else torch.float32,
            memory_format=torch.channels_last,
        )
        inv_scale_y = (1.0 / scale_y) if scale_y > 0.0 else 0.0
        inv_scale_x = (1.0 / scale_x) if scale_x > 0.0 else 0.0

        grid = (N * H_in * W_in,)

        def _launch():
            _bilinear_nhwc_bwd_kernel[grid](
                go_nhwc, gi,
                N, H_in, W_in, H_out, W_out,
                sy_bits, sx_bits,
                inv_scale_y, inv_scale_x,
                C=C,
                IS_BF16=is_bf16,
                num_warps=n_warps,
            )

        _preserve_py_random(_launch)
        return gi, None, None


def triton_upsample_bilinear_nhwc(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return _TritonBilinearNHWC.apply(x, int(size[0]), int(size[1]))
