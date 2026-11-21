# Baseline Measurements (`facebookresearch/vggt @ a288dd0f`)

## Environment & Hardware

- **GPU**: 1 × NVIDIA A100-SXM4-80GB (108 SMs, 80 GiB HBM2e, nominal peak BW 2.04 TB/s, 400W TDP)
- **Host**: 12 vCPU, 236 GB SSD (`/content`)
- **Software**: Python 3.13, PyTorch `2.14.1+cu130`, CUDA 13.0
- **Upstream commit**: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- **Frozen judge `evaluator_sha256`**: `c5b59bf55f251eca2f3571e0ecba04e39fe9ba32f63fd8fd945e0d1f182c966d`

## Frozen Baseline Timing (`task/timing_ref.json`)

Three independent cold-cache runs (`warmup=6`, `window=15`, `repeats=2`, 30 timed iterations per run):

| Run | Window 0 (s) | Window 1 (s) | Total (s) | s / iter | Peak Alloc (GiB) | RNG match |
|---|---:|---:|---:|---:|---:|---|
| `ref_a` | — | — | 57.667 | 1.9222 | 28.04 | True |
| `ref_b` | — | — | 57.636 | 1.9212 | 28.04 | True |
| `ref_c` | — | — | 57.656 | 1.9219 | 28.04 | True |
| **Frozen Mean** | — | — | **57.653** | **1.9218** | **28.04** | **True** |

- Run-to-run spread: **`0.05%`** (`0.0005`), window ratio `0.9067`.
- Deterministic 2-step self-repeatability (`bench/records/ref` vs `runs/ref_repeat`): **3,776 / 3,776 tensors bitwise identical (`PASS`, `N0`, `bitwise_trusted: true`)**.

## Baseline Profile (`runs/baseline_profile`, 5 iterations after 6 warmup)

- **Profiled span**: `8.90 s` (`8,554.7 ms` kernel span), **GPU busy**: `5,820.2 ms` (**68.0%**), **GPU idle / gaps**: `2,734.4 ms` (**32.0%**, 72,853 kernels, 9 kernels/ms).

### GPU Kernel Time by Category (5 iterations, rank 0)

| Category | GPU ms | Share of GPU Busy |
|---|---:|---:|
| `gemm` | 1,895.0 | 31.0% |
| `memcpy/cast` | 1,147.2 | 18.8% |
| `elementwise/reduce` | 1,144.1 | 18.7% |
| `attention` | 911.6 | 14.9% |
| `norm` | 635.2 | 10.4% |
| `conv` | 183.8 | 3.0% |
| `optimizer` | 72.5 | 1.2% |
| `other` | 10.0 | 0.2% |

### Top Kernel Classes & Utilization Flags (`kernels.md`, 5 iterations)

| Kernel Class | Calls | Total ms | % GPU | Median us | Blocks | SM Cov | Flags | Hotspot Attribution |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `pytorch_flash::flash_fwd_kernel` (bf16) | 720 | 904.2 | 15.5% | 375.0 | 1728 | 1.00 | `skewed` | Aggregator (24 DINOv2 + 48 AA blocks) |
| `ampere_bf16_s16816gemm` (128×128 + 128×256) | 2880 | 973.9 | 16.8% | 319–416 | 388–2328 | 1.00 | — | Aggregator QKV/Proj/MLP GEMMs |
| `vectorized_layer_norm_kernel<float>` | 2908 | 635.2 | 10.9% | 91.5 | 16284 | 0.88 | `skewed` | Aggregator `norm1`/`norm2`/`q_norm`/`k_norm` (FP32) + DPT/Camera norms |
| `CUDAFunctor_add<float>` + `BinaryFunctor` mul/add | 14417 | 692.6 | 11.9% | 8–90 | 2423–25488 | 0.62–1.00 | `skewed` | RoPE, LayerScale, residual adds, loss elementwise |
| `Memcpy HtoD (Pageable -> Device)` | 70 | 278.9 | 4.6% | — | — | — | pageable | `pin_memory=False` + CPU `_process_batch` |
| `gatherKthValue<float, int, 1>` | 20 | 226.0 | 3.9% | 11,460 | **1** | **0.01** | **`under-filled`** | `loss.py:torch_quantile` (`torch.kthvalue` on millions of pixels, 1 SM!) |
| `cudnn::convertTensor_kernel<float>` | 2746 | 183.8 | 3.2% | 11.4 | 4096 | 1.00 | `skewed` | NCHW↔NHWC transposes inside `DPTHead` |
| `bfloat16_copy_kernel_cuda` | 8190 | 166.2 | 2.9% | 13.5 | 3072 | 0.65 | — | Per-step FP32→BF16 casting of frozen `aggregator` weights & activations |
| `CatArrayBatchedCopy` + `direct_copy_kernel` + `neg_kernel` + `_scatter_gather` | 13376 | 468.3 | 8.0% | 7–78 | 399–25488 | 0.86–1.00 | `skewed` | Unfused 2D RoPE (`chunk`/`cat`/`neg`/`F.embedding`) + head splits |
| `sm80_xmma_*` / `cutlass_tensorop_s1688*` (TF32 convs) | 1082 | 665.2 | 11.4% | 156–7661 | 81–8612 | 0.69–1.00 | `skewed` | `DPTHead` FP32/TF32 forward + backward convolutions |
| `upsample_bilinear2d_nhwc` (fwd + bwd) | 200 | 149.6 | 2.6% | 275–287 | 22496 | 1.00 | `skewed` | `DPTHead` bilinear upsampling |
| `indexing_backward_kernel_stride_1<float>` | 50 | 61.3 | 1.1% | 1247 | 506634 | 1.00 | — | Boolean mask indexing `pred[mask]`, `loss[quantile_mask]` in `loss.py` |
| `reduce_kernel<512, 1, MaxNanFunctor<long>>` | 960 | 12.5 | 0.2% | 12.8 | **1** | **0.01** | **`under-filled`** | `int(positions.max()) + 1` in `rope.py` (960 GPU→CPU syncs per 5 iters!) |

### Top CPU Bottlenecks (5 iterations)

- `aten::ne` (247 ms), `aten::eq` (244 ms), `aten::any` (216 ms), `aten::abs` (102 ms), `aten::linalg_vector_norm` (94 ms), `aten::div` (57 ms): driven by `normalize_camera_extrinsics_and_points_batch` executing on CPU float32 tensors + GPU→CPU syncs in `rope.py` (`int(positions.max())`), `check_and_fix_inf_nan`, and per-step `.item()` calls.
