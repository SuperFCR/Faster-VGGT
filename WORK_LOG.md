# WORK_LOG — VGGT Training Optimization (ResearchBoost-Agent)

- **Target repository**: `facebookresearch/vggt` @ `a288dd0f14786c93483e45524328726ab7b1b4ce`
- **Methodology & frozen `numerics/`**: `DAGroup-PKU/ResearchBoost-Agent` @ `0c1c745c6e8c2b689781d424860988d1d328aee2`
- **Hardware**: 1 × NVIDIA A100-SXM4-80GB (108 SMs, 80 GiB HBM2e, 2.04 TB/s nominal peak BW), 12 vCPU, Python 3.13.16, PyTorch `2.14.1+cu130`, Triton `3.8.0`, CUDA 13.0
- **Immutable judge `evaluator_sha256`**: `c5b59bf55f251eca2f3571e0ecba04e39fe9ba32f63fd8fd945e0d1f182c966d`
- **Final source digest (`commit 36b1354`)**: `09c8e56856419bb685f14bc1f0a7bc46416eadcc89bfc27b60949f51d2e71dfe`

---

## Summary of Certified Results

| Configuration | Precision Tier | Mode | Ref (s/iter) | Cand (s/iter) | Speedup | Peak Alloc (GiB) | Max \|d\|/σ (bound 3.0σ) | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| Upstream default / switches OFF | N0 | `check --quick` | 1.9218 | — | 1.0000× | 28.04 | 0.000σ (3776/3776 bitwise) | **PASS** |
| Pure N0 subset (`configs/r3_n0.json`) | N0 | `check --quick` | 1.9218 | — | ~1.19× | 29.73 | 0.000σ (3776/3776 bitwise) | **PASS** |
| Pure Lossless N1 (`configs/final_n1.json`) | N1 | **`check --final` (A-B-A)** | 1.9283 | 1.0351 | **1.8630×** | 30.48 | 0.0642σ | **PASS** |
| **Headline (`configs/final.json`)** | **N2** (`DPTHead` BF16; FP32 final proj/act/loss/CameraHead/optim) | **`check --final` (A-B-A)** | **1.9250** | **0.8807** | **2.1857×** | **19.15 (−31.7%)** | **0.0642σ** | **PASS** |

---

## Round 0: Formulation, Frozen Baseline & 6/6 Judge Acceptance

### Baseline Measurements (`task/timing_ref.json`, `results/profiles/baseline_{profile,kernels}.md`)
- Three independent cold-cache timing runs (`warmup=6`, `window=15`, `repeats=2` = 30 timed iterations per run): `[57.667, 57.636, 57.656] s` → mean **`1.9218 s/iter`**, spread **`0.05%`**, window ratio `0.9067`, peak allocated VRAM **`28.04 GiB`**, `rng_identical_across_runs = true`.
- Deterministic 2-step self-repeatability (`train_2step`): **`3,776 / 3,776` tensors bitwise identical (`PASS`, `N0`, `bitwise_trusted=true`)**.
- Frozen 6-seed trajectory σ (`bench/traj/frozen.json`, seeds 42–47, 20 steps): 7 endpoints (`objective`, `loss_camera`, `loss_conf_depth`, `loss_reg_depth`, `loss_grad_depth` on `[10, 20)`; `step_delta_top1_camera` σ=0.16% and `step_delta_top1_depth` σ=0.22% on `[1, 2)`).
- **6/6 Acceptance Suite (`results/acceptance_summary.json`)**:
  - `a_quick_bitwise`: `PASS` (`N0`, `bitwise=true`)
  - `a_full_no_benefit`: `UNAVAILABLE` (`no_benefit`, speedup `0.9974`)
  - `b_bad_setup_error`: `ERROR` (`setup_guard`)
  - `c_bad_semantics_incorrect`: `INCORRECT` (`dead_gradient` / `structure`)
  - `d_equiv_pass` (`AdamW(foreach=False)`): `PASS` (`equivalent`, max `|d|/σ = 0.025σ`, step deltas `0.000σ`)
  - `e_control_arm_incorrect` (`lr * 1.02`): `INCORRECT` (`different`, `step_delta_top1_camera = +12.18σ`, `step_delta_top1_depth = +9.24σ`)
  - `f_numerics_selftest`: `ok=true` (12/12 mutations caught)

### Baseline Profile Findings (5 iters after 6 warmup)
- Span `8.55 s`, **GPU busy `5.82 s` (68.0%)**, idle/host gaps `2.73 s` (32.0%), `72,853` kernels (`9/ms`).
- Dominant bottlenecks:
  1. Host/CPU gaps (`2.73 s`): CPU `_process_batch` (`normalize_camera_extrinsics_and_points_batch`), pageable `Memcpy HtoD` (`278.9 ms`), 960× `int(positions.max())` syncs in `rope.py` (`MaxNanFunctor`, SM cov `0.01`), `check_and_fix_inf_nan` syncs, per-step `.item()` syncs.
  2. Frozen aggregator memory-bound overhead (`~1,790 ms`): FP32 `vectorized_layer_norm_kernel` (`635 ms`), unfused RoPE & LayerScale elementwise ops (`1,161 ms`), per-step FP32→BF16 weight casting (`8,190` calls / `166 ms`).
  3. Loss single-SM quantile & boolean compaction (`~287 ms`): `gatherKthValue<float, int, 1>` (`226.0 ms`, **1 block / 108 SMs, SM cov `0.01`**), `indexing_backward_kernel_stride_1` (`61.3 ms`).
  4. `DPTHead` layout & FP32/TF32 convolutions (`~1,000 ms`): `cudnn::convertTensor_kernel` (`183.8 ms`, `2,746` calls), TF32 convs (`665 ms`), `upsample_bilinear2d` (`150 ms`), repeated FP64 UV `pos_embed` recomputation, `frames_chunk_size=8`.
  5. Optimizer & clipping (`72.5 ms`): unfused `AdamW` + per-group `.item()` clipping syncs.

---

## Round 1: Host Pipeline, Sync Removal, GPU Normalization & 1-Frame RoPE Cache (`cd6cb48`)

### Changes (all OFF by default)
- `VGGT_OPT_PIN_MEM=1` (`training/data/dynamic_dataloader.py`): `pin_memory=True` and `persistent_workers=True` (when `num_workers > 0`).
- `VGGT_OPT_NO_SYNC=1` (`training/train_utils/general.py`, `training/trainer.py`): branchless `torch.nan_to_num` in `check_and_fix_inf_nan`, single batched D2H flush (`_flush_deferred_logs`) at end of training step.
- `VGGT_OPT_ROPE_CACHE=1` (`vggt/layers/rope.py`, `vggt/models/aggregator.py`): compute 1-frame position grid & share cos/sin tables across all 48 AA blocks without `int(positions.max())` GPU→CPU sync.
- `VGGT_OPT_GPU_NORM=1` (`training/trainer.py`, `training/train_utils/normalization.py`): move batch HtoD first (`non_blocking=True`) and run `normalize_camera_extrinsics_and_points_batch` on GPU.

### Results (`bench.py time` & `check --quick`)
- `r1_pin`: `1.9036 s/iter` (`1.0096×`)
- `r1_ropecache`: `1.8764 s/iter` (`1.0242×`)
- `r1_nosync`: `1.7525 s/iter` (`1.0966×`)
- `r1_gpunorm`: `1.6767 s/iter` (`1.1462×`)
- `r1_n0` (`PIN_MEM + NO_SYNC + ROPE_CACHE`): `1.6545 s/iter` (**`1.1615×`**), `check --quick`: **PASS, `3,776/3,776` bitwise (`N0`)**.
- `r1_all` (`+ GPU_NORM`): **`1.5534 s/iter` (`1.2372×`)**, `check --quick`: **PASS (`N1`, `0` exact failures, `0` dtype changes)**, peak alloc `28.04 GiB`.
- Post-Round-1 profile (`results/profiles/r1_all_{profile,kernels}.md`): GPU busy rose `68.0% → 86.0%` (`5.93 s / 6.89 s`); `Memcpy HtoD` dropped `278.9 → 98.3 ms`; `MaxNanFunctor` (960 calls) and `_scatter_gather` (3,840 calls) eliminated.

---

## Round 2: Frozen Aggregator BF16 Weight Cache & Custom Triton Kernels (`e646293`)

### Profile Diagnosis (from `r1_all`)
- Frozen aggregator non-GEMM/non-FlashAttention overhead dominated remaining GPU time (~1,790 ms / 30%): `vectorized_layer_norm_kernel` (`638 ms`), `CUDAFunctor_add` + `BinaryFunctor` (`694 ms`), `CatArrayBatchedCopy` + `direct_copy_kernel` + `neg_kernel` (`440 ms`), `bfloat16_copy_kernel_cuda` (`167 ms`, `8,190` calls).

### Changes
- `VGGT_OPT_AGG_BF16_CACHE=1` (`vggt/layers/triton_ops.py`, `attention.py`, `mlp.py`, `patch_embed.py`): cache BF16 copies of frozen (`requires_grad=False`) `nn.Linear` and `nn.Conv2d` weights/biases keyed by tensor `_version` (not registered in `state_dict`/parameters; 100% N0 bitwise identical to autocast).
- `VGGT_OPT_TRITON_ROPE=1` (`vggt/layers/triton_ops.py`, `rope.py`, `attention.py`): `_fused_qkv_qknorm_rope_kernel` fuses QKV unpack + exact FP32 QK-Norm + exact FP32 2D RoPE (modulo 1-frame `P` tables) → BF16 `Q, K` directly for FlashAttention, with zero-copy strided BF16 view for `V` and skipped on DINOv2 blocks (`q_norm=Identity, rope=None`). Wraps launches with `_preserve_py_random` so Triton cache never perturbs Python's global `random`.
- `VGGT_OPT_TRITON_NORM=1` (`vggt/layers/triton_ops.py`, `block.py`): `_layernorm_to_bf16_kernel`, `_ls_add_layernorm_bf16_kernel`, and in-place `_ls_add_kernel` (for freshly allocated `x_mid`) keeping exact FP32 residual stream while avoiding intermediate FP32→BF16 casts.
- **Rejected attempt (logged)**: tested `torch.ops.aten._addmm_activation(..., use_gelu=True)` to fuse `Mlp.fc1 + GELU` (`92 ms`); unit test showed cuBLASLt uses tanh-approx GELU (`0.0` vs tanh, `4.73e-4` vs exact erf-GELU) — **Tier N3 (forbidden)**, immediately rejected. Exact `GeluCUDAKernelImpl` already achieves ~99% of A100's 2.04 TB/s HBM BW.

### Results (`bench.py time` & `check --quick`)
- `r2_n0` (`r1_n0 + AGG_BF16_CACHE`): `check --quick`: **PASS, `3,776/3,776` bitwise (`N0`)**.
- `r2_bf16cache`: `1.5336 s/iter` (`1.2531×`)
- `r2_triton_norm`: `1.4582 s/iter` (`1.3179×`)
- `r2_triton_rope`: `1.2983 s/iter` (`1.4802×`)
- `r2_all`: **`1.2377 s/iter` (`1.5527×`)**, `check --quick`: **PASS (`N1`, `0` exact failures, `0` dtype changes)**, peak alloc `29.73 GiB`.
- Post-Round-2 profile (`results/profiles/r2_all_{profile,kernels}.md`): total GPU time dropped `5.93 s → 4.49 s` (`45,333` kernels down from `72,853`); ~1,790 ms of aggregator overhead compressed into `190.2 ms` of 4 Triton kernels (**9.4× faster**).

---

## Round 3: Multi-SM Exact Triton Radix-Select Quantile & Sync-Free Masked Loss (`e9cf0c5`)

### Profile Diagnosis (from `r2_all`)
- `multitask_loss` became the #1 non-core hotspot (`669.4 ms` GPU, `1,117.7 ms` CPU):
  - `gatherKthValue<float, int, 1>`: **`223.2 ms`** (20 calls, **1 block / 108 SMs, `SM cov = 0.01`**, `under-filled`).
  - `indexing_backward_kernel_stride_1<float>`: **`61.7 ms`** (50 calls) + `aten::nonzero` (260 calls) from boolean compaction `gt[mask] - pred[mask]`, `conf[mask]`, `loss_tensor[quantile_mask]`, plus 12+ GPU→CPU syncs per microbatch.

### Changes
- `VGGT_OPT_FAST_QUANTILE=1` (`training/train_utils/fast_quantile.py`, `training/loss.py`): 4-pass 8-bit Triton histogram radix-select over order-preserving IEEE-754 float32→int32 keys across `216` persistent CTAs (2 CTAs/SM), with exact IEEE-754 `fp64` `q` passed via `int64` bitcast (`q_bits_i64`) and GPU-side prefix scan. Microbench on 12M float32: **`82.80 ms → 0.52 ms` (`160.2×` faster, 100% bitwise identical `N0`)**.
- `VGGT_OPT_MASKED_LOSS=1` (`training/loss.py`): dense branchless `_regression_loss_masked`, masked `fast_quantile_1d(..., mask=mask)`, sync-free `gradient_loss` and `compute_camera_loss` (zero boolean compaction, zero `indexing_backward_kernel_stride_1`).

### Results (`bench.py time` & `check --quick`)
- `r3_n0` (`r2_n0 + FAST_QUANTILE`): `check --quick`: **PASS, `3,776/3,776` bitwise (`N0`)**.
- `r3_fast_quantile`: `1.2043 s/iter` (`1.5957×`)
- `r3_masked_loss`: `1.1669 s/iter` (`1.6469×`)
- `r3_all`: **`1.1655 s/iter` (`1.6489×`)**, `check --quick`: **PASS (`N1`, `0` exact failures, `0` dtype changes)**.
- Post-Round-3 profile (`results/profiles/r3_all_{profile,kernels}.md`): `multitask_loss` GPU dropped `669.4 → 209.3 ms` (**3.2×**); `gatherKthValue` and `indexing_backward_kernel_stride_1` completely eliminated.

---

## Round 4: `DPTHead` Layout, Pos-Embed Cache, Frame Chunking, `FusedAdamW` & NH RoPE Layout (`b9d430a`)

### Profile Diagnosis (from `r3_all`)
- `depth_head` (`1,212.9 ms` GPU): `cudnn::convertTensor_kernel` (`184.8 ms`, `2,746` calls), TF32 convs (`723 ms`), `upsample_bilinear2d` (`152 ms`), repeated FP64 UV `pos_embed` recomputation, `frames_chunk_size=8` splitting 24-frame sequences into 3 passes.
- `aggregator`: `480` calls to `direct_copy_kernel` (`33.7 ms`) after FlashAttention because `Q, K` were allocated in `(B, H, N, 64)` contiguous instead of `(B, N, H, 64)` transposed layout.
- `optimizer`: `72.5 ms` unfused AdamW + 2 `GradientClipper` `.item()` syncs per step.

### Changes
- `VGGT_OPT_DPT_POS_CACHE=1` (`vggt/heads/dpt_head.py`): cache `_apply_pos_embed` by `(patch_w, patch_h, W, H, C, ratio, dtype, device, is_cl)` (N0 bitwise).
- `VGGT_OPT_DPT_CHANNELS_LAST=1` (`vggt/heads/dpt_head.py`): convert `DPTHead` conv weights to `torch.channels_last` in `__init__` (before DDP/optimizer/load_state_dict), use zero-copy NHWC view `x.view(N, patch_h, patch_w, C).permute(0, 3, 1, 2)` at entry, and zero-copy NHWC view in `activate_head`.
- `VGGT_OPT_DPT_CHUNK=0` (`vggt/heads/dpt_head.py`): process all frames in a single pass (`frames_chunk_size=None`), preserving `channels_last` through `custom_interpolate`.
- `VGGT_OPT_FUSED_ADAMW=1` (`training/train_utils/optimizer.py`, `gradient_clip.py`, `trainer.py`): `AdamW(..., fused=True)` + defer `GradientClipper` `.item()` syncs into `_flush_deferred_logs`.
- `VGGT_OPT_TRITON_ROPE_NH=1` (`vggt/layers/triton_ops.py`): write `Q, K` in `(B, N, H, 64).transpose(1, 2)` physical layout so FlashAttention returns `(B, N, H, 64)` and `x.transpose(1, 2).reshape(B, N, C)` becomes zero-copy.

### Results (`bench.py time` & `check --quick`)
- `r4_poscache`: `1.1634 s/iter` (`1.6519×`)
- `r4_cl`: `1.1587 s/iter` (`1.6585×`)
- `r4_cl_chunk16`: `1.1540 s/iter` (`1.6653×`)
- `r4_cl_chunk0`: `1.1435 s/iter` (`1.6806×`)
- `r4_all_adamw`: `1.1328 s/iter` (`1.6965×`)
- `r4_rope_nh`: **`1.1319 s/iter` (`1.6979×`)**, `check --quick`: **PASS (`N1`)**, peak alloc `30.48 GiB`.

---

## Round 5 & Round 6: Hidden Sync Elimination, Cross-Block Triton Fusion, Local BF16 `DPTHead`, Atomic-Free Triton Bilinear Upsample & Zero-Recompile Specialization (`d95dfb3`, `36b1354`)

### Root-Cause Discoveries & Failed/Regression Attempts (Logged per Methodology)
1. **Hidden syncs inside `multitask_loss`** (`r5_n1_all` initial CPU time `1,516 ms` vs `195 ms` GPU):
   - Found and eliminated 5 hidden GPU→CPU / pageable-H2D syncs per microbatch via `torch.cuda.set_sync_debug_mode("warn")`:
     - `fast_quantile.py`: `state = torch.tensor([0, k_init], device=cuda)` pageable H2D sync → replaced with GPU-allocated `torch.empty(2)` + kernel scalar `k_init_i32`.
     - `vggt/utils/rotation.py`: `torch.tensor(0.1).to(device)`, `quat_candidates[one_hot > 0.5]`, `ret[positive_mask] = sqrt(x[positive_mask])`, and `out[..., [1, 2, 3, 0]]` → replaced with `.clamp(min=0.1)`, `.gather()`, `torch.where()`, and `torch.roll(out, -1, -1)` (0 sync warnings, 100% N0 bitwise identical).
   - Pure N1 jumped `1.7110× → 1.7755×` (`1.0824 s/iter`), and `multitask_loss` GPU dropped to `42.5 ms` (**15.8×** vs baseline!).
2. **Failed attempt #1 — Naive `VGGT_OPT_HEAD_BF16=all` (`1.7070×`, `r6_bf16_all`)**:
   - Profiling (`runs/r6/profile_r6_bf16_all`) exposed THREE distinct regressions:
     1. `DPTHead.scratch.output_conv2[2]` (`Conv2d(32, 2, 1x1)`) in BF16 had `C_out=2` (`align2`) and fell back to a pathological 1-CTA `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_32x1_nt_align2` wgrad kernel taking **`622.3 ms`** (15.5% of total GPU time!). **Fix**: run `output_conv2[2]` in FP32 via `F.linear` on the zero-copy NHWC view right before `activate_head` (eliminates `622 ms` AND improves numerical precision!).
     2. `CameraHead` in BF16 was slower (`99.9 ms` vs `80.3 ms` FP32) because casting 202M FP32 weights to BF16 per microbatch outweighed tiny-batch (`B*S <= 48`) GEMM savings. **Fix**: set `VGGT_OPT_HEAD_BF16=depth` (keep `CameraHead` 100% FP32).
     3. `F.interpolate(..., mode='bilinear')` upcast BF16→FP32→BF16 (`153 ms` + `109 ms` casts).
3. **Failed attempt #2 — Forcing PyTorch `F.interpolate` in BF16 (`1.5412×`, `r6_bf16_depth_v2`)**:
   - Profiling showed PyTorch's `upsample_bilinear2d_backward_nhwc_out_frame<BFloat16>` uses a slow CAS-loop BF16 `atomicAdd` taking **`1,486.9 ms`** (31.2% of GPU time, **20× slower** than FP32 `73.7 ms`!).
   - **Fix (`VGGT_OPT_TRITON_UPSAMPLE=1`)**: wrote a deterministic, atomic-free gather-based Triton NHWC bilinear upsample kernel (`_bilinear_nhwc_fwd_kernel` + `_bilinear_nhwc_bwd_kernel` with exact `fp64` source coordinates) in `vggt/layers/triton_ops.py`. `DPTHead` fwd+bwd microbench dropped `337 ms (FP32) → 192 ms (BF16+FP32up) → 158.5 ms (BF16+TritonUp)`!
4. **Failed attempt #3 — Initial Triton upsample bench regression (`1.8842×` vs `2.0128×` FP32up)**:
   - Profile (`host time per iteration: 3.173, 0.652, 0.755, 0.930, 0.621 s`) caught **Triton JIT recompilation inside the timed window**! Triton's default `% 16` divisibility specialization on dynamic `(N, H_in, W_in, H_out, W_out, P)` triggered 2.5 s of cold JIT compiles whenever VKitti sampled a new aspect ratio.
   - **Fix**: added `do_not_specialize=[...]` for dynamic dimensions across all 8 Triton kernels in `triton_ops.py` and `fast_quantile.py`, while preserving 128-bit vectorization on fixed channel strides (`C=1024`, `stride_qkv_n=3072`). Verified 0 recompiles across arbitrary prime/odd/div16 shapes (`0.5–0.7 ms`).
5. **Cross-block fusion (`VGGT_OPT_TRITON_FUSED_BLK=1`)**:
   - Fused Block `i`'s `ls2_add` with Block `i+1`'s `norm1 -> bf16` across 23 DINOv2 + 47 AA block boundaries (`700` fewer kernel launches & FP32 HBM reads per 5 iters).
6. **Validation-path fix caught during E2E wrap-up (`36b1354`)**:
   - Scoped `_update_and_log_scalars` deferral to `phase == "train"` so `Trainer.val_epoch` flushes validation scalar meters immediately.

### Results (`bench.py time` after `do_not_specialize`)
- Pure N1 (`r5_n1_all`): `1.0476 s/iter` (`1.8344×`)
- Pure N1 + Triton Upsample (`r5_n1_upsample`): **`1.0347 s/iter` (`1.8574×`)**
- N2 + FP32 Upsample (`r6_bf16_depth_fp32up`): `0.9202 s/iter` (`2.0884×`)
- N2 + Triton Upsample (`r6_bf16_depth_tritonup`): **`0.8752 s/iter` (`2.1959×`)**, peak alloc **`19.15 GiB`** (−31.7% vs baseline!).

---

## Remaining Kernel Audit (`results/profiles/r6_final_{profile,kernels}.md`)

- Profile span (5 iters): **`4.07 s`** (down from **`8.55 s`** baseline), **GPU busy `3.31 s` (81%)** (up from 68%, 43% less total GPU work).
- **72.2% (`2,389 ms / 3,310 ms`) of remaining GPU time is irreducible core compute**:
  - `flash_fwd_kernel` (frozen aggregator): `946.7 ms` (28.6%)
  - `ampere_bf16_s16816gemm` (frozen aggregator): `1,012.0 ms` (30.6%)
  - `DPTHead` BF16 Tensor Core convs: `~340 ms` (10.3%)
  - `CameraHead` FP32/TF32 GEMMs: `~90 ms` (2.7%)
- Remaining non-core kernels (`920 ms` total) are HBM-bandwidth-bound at hardware limits or tiny residual ops:
  - `_ls_add_layernorm_bf16_kernel` (`130.0 ms`, `1,420` calls): minimal FP32 residual stream bandwidth.
  - `Memcpy HtoD (Pinned)` (`98.3 ms`, `40` calls): PCIe bandwidth of input tensors.
  - `GeluCUDAKernelImpl` (`93.1 ms`, `720` calls): ~99% of A100 2.04 TB/s peak BW; cuBLASLt fusion uses N3 tanh-approx GELU (forbidden).
  - `_bilinear_nhwc_{fwd,bwd}_kernel` (`83.4 ms`, `100` calls): atomic-free deterministic BF16 NHWC gather.
  - `_fused_qkv_qknorm_rope_kernel` (`~37 ms`, `480` calls): single-pass FP32 QK-Norm+RoPE with zero-copy V & FlashAttention NH layout.
  - `FusedOptimizer` `multi_tensor_apply_kernel` (`21.2 ms`, `10` calls): single fused AdamW pass.
- Plateau legitimately declared above the 2.0× target!

---

## Formal A-B-A Final Verification & Real End-to-End Validation (`36b1354`)

### 1. Formal Harness `check --final` (commit `36b1354`, `source_digest = 09c8e568...71dfe`)
- **`results/final_check.json` (`configs/final.json`, Headline N2)**:
  - **`status: PASS`**, `final: true`, `evaluator_sha256: c5b59bf55f251eca2f3571e0ecba04e39fe9ba32f63fd8fd945e0d1f182c966d`
  - A-B-A timing: `ref_a = 1.9238 s/iter`, `cand = 0.8807 s/iter`, `ref_b = 1.9262 s/iter` (`ref_mean = 1.9250 s/iter`, spread `0.13%`) → **`speedup = 2.1857×`** (`bar = 1.02`, `problems = []`, `fingerprint_diffs = []`).
  - Peak allocated VRAM: **`19.15 GiB`** vs **`28.04 GiB`** baseline (**−31.7%**).
  - Gate 2: `0` errors, `0` exact failures, `0` dtype changes, `0` precision_mode changes, `2,703/3,776` bitwise identical.
  - Gate 3 (20-step trajectory, `3.0σ` bound): `verdict = equivalent` — `objective: +0.0214σ`, `loss_camera: -0.0642σ`, `loss_conf_depth: +0.0267σ`, `loss_reg_depth: +0.0110σ`, `loss_grad_depth: +0.0282σ`, `step_delta_top1_camera: -0.0521σ`, `step_delta_top1_depth: +0.0403σ`.
- **`results/final_check_n1.json` (`configs/final_n1.json`, Pure Lossless N1)**:
  - **`status: PASS`**, `final: true`, `speedup = 1.8630×` (`ref_mean = 1.9283 s/iter` [`1.9252 / 1.9314`, spread `0.32%`] → `cand = 1.0351 s/iter`), max `|d|/σ = 0.0642σ`.
- **`results/final_check_default_off.json` (No switches)**:
  - **`status: PASS`**, `N0`, **`3,776/3,776` bitwise identical**, `0` exact failures, `0` dtype changes.

### 2. Real End-to-End Training, Checkpoint Save/Reload & Held-Out Validation (`results/e2e_summary.json`)
- Executed unmodified upstream `training/launch.py` (`torchrun --nproc-per-node=1`) for 1 full epoch (`limit_train_batches=40` → 41 steps / 82 microbatches, real `checkpoint_0.pt` save at `save_freq=1`, `strict=True` reload into fresh `VGGT`, and held-out VKitti `split=test` validation epoch `limit_val_batches=15`, `val_len_test=100`):
  - **Train avg losses**: `objective: 0.0344 vs 0.0344`, `loss_camera: 0.0049 vs 0.0049`, `loss_conf_depth: -0.0589 vs -0.0590`, `loss_reg_depth: 0.0494 vs 0.0493`, `loss_grad_depth: 0.0537 vs 0.0539`.
  - **Held-out val avg losses**: `loss_camera: 0.0039 vs 0.0039`, `loss_conf_depth: -0.0542 vs -0.0560`, `loss_reg_depth: 0.0591 vs 0.0590`, `loss_grad_depth: 0.0464 vs 0.0466`.
  - **Checkpoints**: `checkpoint_0.pt` (`6.17 GiB`, `1,341` keys, `0` non-finite tensors, `strict_reload_ok=true`) for both baseline and optimized.
  - **Wall clock**: `192.74 s → 145.01 s` (`1.33×` full wall including startup, 5 GiB checkpoint load, cold Triton compile, 6.17 GiB checkpoint save & validation; steady-state step 40 batch time `2.8256 s → 1.2702 s` = **`2.22×`**, peak memory `32.0 GB → 20.0 GB` train / `19.0 GB` val).
