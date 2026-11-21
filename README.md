# vggt-ResearchBoost-Agent

Training-step acceleration of **[VGGT (`facebookresearch/vggt`)](https://github.com/facebookresearch/vggt)** strictly following the official frozen-judge protocol, methodology, and unmodified `numerics/` component of **[ResearchBoost-Agent (`DAGroup-PKU/ResearchBoost-Agent`)](https://github.com/DAGroup-PKU/ResearchBoost-Agent)**.

---

## Headline Results

**Upstream `facebookresearch/vggt` @ `a288dd0f` · `training/config/default.yaml` fine-tuning (`VGGT-1B`, frozen 908M aggregator + trainable 202M `camera_head` + 12M `depth_head`) · Virtual KITTI 2.0.3 (`max_img_per_gpu=48`, `accum_steps=2`)**

| Item | Headline (`configs/final.json`, Tier N2) | Pure Lossless (`configs/final_n1.json`, Tier N1) |
|---|---|---|
| **Speedup (`check --final` A-B-A)** | **2.1857×** (`1.9250 → 0.8807 s/iter`) | **1.8630×** (`1.9283 → 1.0351 s/iter`) |
| **Peak Allocated VRAM** | **19.15 GiB** vs `28.04 GiB` (**−31.7%**) | `30.48 GiB` vs `28.04 GiB` |
| **Hardware** | 1 × NVIDIA A100-SXM4-80GB (PyTorch `2.14.1+cu130`, Triton `3.8.0`, CUDA 13.0) | Same |
| **Measurement scope** | Fresh cold-cache `torchrun` A-B-A (`ref_a → cand → ref_b`), `warmup=6`, `2 × 15 = 30` timed iters, `ref_spread = 0.13%` | Same (`ref_spread = 0.32%`) |
| **Judge status** | **`PASS` (`final: true`)** — Gate 0/1/2/3 all passed | **`PASS` (`final: true`)** — Gate 0/1/2/3 all passed |
| **Gate 2 (2-step deterministic)** | `0` errors, `0` exact failures, `0` dtype changes, `2,703/3,776` bitwise | `0` errors, `0` exact failures, `0` dtype changes, `2,704/3,776` bitwise |
| **Gate 3 (20-step 3σ trajectory)** | `equivalent`, **max \|d\|/σ = 0.0642σ** (bound `3.0σ`) | `equivalent`, **max \|d\|/σ = 0.0642σ** (bound `3.0σ`) |
| **Default-off parity** | **`PASS` (`N0`)** — `3,776 / 3,776` tensors bitwise identical | Same |
| **Frozen judge `evaluator_sha256`** | `c5b59bf55f251eca2f3571e0ecba04e39fe9ba32f63fd8fd945e0d1f182c966d` | Same |
| **Certified `source_digest`** | `09c8e56856419bb685f14bc1f0a7bc46416eadcc89bfc27b60949f51d2e71dfe` | Same |

> **Official compliance**: All optimizations sit behind explicit environment-variable switches (OFF by default). The frozen judge (`bench/`, `numerics/`, `task/`) passed all **6/6** negative & control-arm self-tests (including rejecting `lr × 1.02` at `+12.18σ / +9.24σ`, catching dead gradients, and 12/12 `numerics` mutations) before optimization began and remained cryptographically immutable throughout (`evaluator_sha256` identical across formulation, every round, and `--final`).

---

## Optimization Breakdown & Ablations

All intermediate timing rows are cold-cache `bench/bench.py time` runs against the frozen baseline (`1.9218 s/iter`, 3-run spread `0.05%`); final rows are official A-B-A `bench/bench.py check --final` runs.

| Round | Config / Switch Added | Tier | s / iter | Speedup | Peak Alloc (GiB) | What Changed |
|---|---|---|---:|---:|---:|---|
| Baseline | Unmodified `vggt @ a288dd0f` | N0 | 1.9218 | 1.0000× | 28.04 | Upstream `training/launch.py` + `default.yaml` |
| R1 | `VGGT_OPT_PIN_MEM=1` | N0 | 1.9036 | 1.0096× | 28.04 | `pin_memory=True`, `persistent_workers=True` |
| R1 | `VGGT_OPT_ROPE_CACHE=1` | N0 | 1.8764 | 1.0242× | 28.04 | 1-frame RoPE cos/sin cache across 48 AA blocks; removes 960× `int(pos.max())` syncs |
| R1 | `VGGT_OPT_NO_SYNC=1` | N0 | 1.7525 | 1.0966× | 28.04 | Branchless `nan_to_num` + single batched end-of-step D2H scalar flush |
| R1 | `VGGT_OPT_GPU_NORM=1` | N1 | 1.6767 | 1.1462× | 28.04 | Non-blocking HtoD before GPU `normalize_camera_extrinsics_and_points_batch` |
| R1 | `r1_n0` (`PIN + NO_SYNC + ROPE_CACHE`) | **N0** | 1.6545 | **1.1615×** | 28.04 | **3,776/3,776 bitwise identical (`PASS N0`)** |
| R1 | `r1_all` (`+ GPU_NORM`) | N1 | 1.5534 | 1.2372× | 28.04 | GPU busy `68% → 86%` |
| R2 | `+ VGGT_OPT_AGG_BF16_CACHE=1` | **N0*** | 1.5336 | 1.2531× | 29.73 | `_version`-keyed BF16 cache for frozen aggregator weights (`r2_n0`: **3,776/3,776 N0**) |
| R2 | `+ VGGT_OPT_TRITON_NORM=1` | N1 | 1.4582 | 1.3179× | 29.73 | Triton `LayerNorm→BF16` and `LayerScale+Add+LayerNorm→BF16` (exact FP32 residual) |
| R2 | `+ VGGT_OPT_TRITON_ROPE=1` | N1 | 1.2983 | 1.4802× | 29.73 | Triton fused QKV unpack + FP32 QK-Norm + 2D RoPE → BF16, zero-copy V |
| R2 | `r2_all` (all R2 stacked) | N1 | 1.2377 | 1.5527× | 29.73 | Aggregator non-core overhead `1,790 ms → 190 ms` (9.4×) |
| R3 | `+ VGGT_OPT_FAST_QUANTILE=1` | **N0*** | 1.2043 | 1.5957× | 29.73 | 4-pass 8-bit Triton radix-select (**160.2×** faster than `kthvalue`; `r3_n0`: **3,776/3,776 N0**) |
| R3 | `+ VGGT_OPT_MASKED_LOSS=1` (`r3_all`) | N1 | 1.1655 | 1.6489× | 29.73 | Dense branchless masked loss; eliminates `indexing_backward` & `nonzero` compaction |
| R4 | `+ DPT_POS_CACHE + CHANNELS_LAST + DPT_CHUNK=0` | N1 | 1.1435 | 1.6806× | 30.48 | Cached UV pos-embed (N0), NHWC `DPTHead`, unchunked frames |
| R4 | `+ VGGT_OPT_FUSED_ADAMW=1 + TRITON_ROPE_NH=1` | N1 | 1.1319 | 1.6979× | 30.48 | Fused AdamW + deferred clip sync + zero-copy FlashAttention `(B, N, H, 64)` output |
| R5 | `+ TRITON_FUSED_BLK=1` + `rotation.py`/quantile zero-sync | N1 | 1.0824 | 1.7755× | 30.48 | Cross-block `ls2+norm1` fusion (70 boundaries) + 0 sync warnings in `multitask_loss` |
| R5 | `+ do_not_specialize` + `TRITON_UPSAMPLE=1` (`final_n1`) | **N1** | **1.0351** | **1.8630×** | 30.48 | Zero dynamic-shape Triton recompiles + atomic-free deterministic NHWC bilinear upsample (**`--final PASS`**) |
| R6 | `+ VGGT_OPT_HEAD_BF16=depth` (**`configs/final.json`**) | **N2** | **0.8807** | **2.1857×** | **19.15** | Local BF16 `DPTHead` convs + FP32 `F.linear` final 32→2 proj/act/loss + FP32 `CameraHead` (**`--final PASS`**) |

*\*Verified `3,776/3,776` bitwise identical (`N0`) when stacked with `r1_n0` (`configs/r2_n0.json`, `configs/r3_n0.json`).*

---

## Kernel Utilization Before vs. After (5 Iterations, Rank 0)

| Metric | Baseline (`baseline_profile.md`) | Final (`r6_final_profile.md`) | Change |
|---|---:|---:|---:|
| Profiled 5-iter window | 8.55 s | **4.07 s** | **−52.4%** |
| GPU busy (kernel time) | 5.82 s (68.0%) | **3.31 s (81.3%)** | **−43.1% GPU work, +13.3 pp util** |
| Kernel launches (5 iters) | 72,853 | **~24,000** | **~3× fewer launches** |
| `multitask_loss` GPU (fwd) | 669.4 ms | **63.0 ms** | **10.6× faster** |
| `gatherKthValue` (SM cov 0.01) | 226.0 ms (20 calls) | **0.0 ms** | Eliminated (Triton radix-select) |
| `cudnn::convertTensor_kernel` | 183.8 ms (2,746 calls) | **0.0 ms** | Eliminated (NHWC + BF16 DPT) |
| `bfloat16_copy_kernel_cuda` | 166.2 ms (8,190 calls) | **9.6 ms** | **−94.2%** |
| `Memcpy HtoD` | 278.9 ms (pageable) | **98.3 ms** (pinned) | **−64.8%** |
| Core Tensor Core / FlashAttn share | ~44% | **72.2%** (`2.39 s / 3.31 s`) | Compute-dominated |

---

## Gate 3 Trajectory Equivalence (`results/final_check.json`)

20-step deterministic trajectory on `seed=42` vs. 6-seed frozen baseline (`bench/traj/frozen.json`, seeds 42–47, bound `±3.0σ`):

| Endpoint | Frozen σ | Headline (`final.json`) \|d\|/σ | Pure N1 (`final_n1.json`) \|d\|/σ | Control Arm (`lr × 1.02`) \|d\|/σ | State |
|---|---:|---:|---:|---:|---|
| `objective` (`[10, 20)`) | 0.03245 | `+0.0214σ` | `+0.0158σ` | `+0.0288σ` | within |
| `loss_camera` (`[10, 20)`) | 0.000589 | `-0.0642σ` | `-0.0642σ` | `+0.1152σ` | within |
| `loss_conf_depth` (`[10, 20)`) | 0.02478 | `+0.0267σ` | `+0.0254σ` | `+0.0199σ` | within |
| `loss_reg_depth` (`[10, 20)`) | 0.00466 | `+0.0110σ` | `+0.0126σ` | `+0.0133σ` | within |
| `loss_grad_depth` (`[10, 20)`) | 0.00611 | `+0.0282σ` | `+0.0024σ` | `+0.0068σ` | within |
| `step_delta_top1_camera` (`[1, 2)`) | 6.54e-8 (0.16%) | `-0.0521σ` | `-0.0521σ` | **`+12.178σ` (REJECT)** | within |
| `step_delta_top1_depth` (`[1, 2)`) | 8.59e-8 (0.22%) | `+0.0403σ` | `-0.0072σ` | **`+9.239σ` (REJECT)** | within |

---

## Real End-to-End Training, Checkpoint Save/Reload & Held-Out Validation

Executed unmodified upstream `training/launch.py` (`torchrun --nproc-per-node=1`) on both `targets/reference` (baseline) and `vggt` (`configs/final.json`) with identical seed (`42`), `max_epochs=1`, `limit_train_batches=40` (41 steps / 82 microbatches), `save_freq=1` (real `checkpoint_0.pt` save & `strict=True` reload into fresh `VGGT`), and `val_epoch_freq=1` on held-out VKitti `split=test` (`results/e2e_summary.json`):

| Metric | Baseline (`targets/reference`) | Optimized (`configs/final.json`) | Delta / Ratio |
|---|---:|---:|---:|
| Full wall clock (startup + 5 GiB load + cold JIT + train + 6.17 GiB save + val) | 192.74 s | **145.01 s** | **1.33×** |
| Steady-state train step 40 `Batch Time` | 2.8256 s | **1.2702 s** | **2.22×** |
| Steady-state val `Batch Time` | 1.9073 s | **1.0178 s** | **1.87×** |
| Peak GPU Memory (train / val) | 29.0 / 32.0 GB | **20.0 / 19.0 GB** | **−9 to −13 GB** |
| Train avg `loss_objective` | 0.0344 | **0.0344** | `0.0000` |
| Train avg `loss_camera` | 0.0049 | **0.0049** | `0.0000` |
| Train avg `loss_conf_depth` / `reg` / `grad` | -0.0589 / 0.0494 / 0.0537 | **-0.0590 / 0.0493 / 0.0539** | `≤ 0.0002` |
| Held-out Val avg `loss_camera` | 0.0039 | **0.0039** | `0.0000` |
| Held-out Val avg `loss_conf_depth` / `reg` / `grad` | -0.0542 / 0.0591 / 0.0464 | **-0.0560 / 0.0590 / 0.0466** | `≤ 0.0018` |
| Saved `checkpoint_0.pt` (`6.17 GiB`, `1,341` keys) | 0 non-finite, `strict=True` OK | **0 non-finite, `strict=True` OK** | Verified |

---

## Repository Structure

- `vggt/`, `training/` — VGGT codebase with all optimizations behind default-off `VGGT_OPT_*` env switches (see `vggt/layers/triton_ops.py`, `training/train_utils/fast_quantile.py`, `WORK_LOG.md`).
- `bench/`, `numerics/` — Frozen ResearchBoost-Agent judge harness (`bench/bench.py`, `bench/recipe.py`, `bench/timing.py`, `bench/vggt_adapter.py`, `bench/kernel_util.py`, frozen `bench/thresholds.json`, `bench/traj/frozen.json`, `bench/records/ref/*.json`, unmodified `numerics/`).
- `configs/` — Switch JSONs for every round (`r1_*` through `r6_*`), `configs/final.json` (headline **2.1857×**), and `configs/final_n1.json` (pure lossless **1.8630×**).
- `FINAL.json` — Pointer to the winning tree and config (`configs/final.json`).
- `GOAL.md`, `task/` — Frozen task contract, baseline breakdown (`task/baseline.md`), and frozen baseline manifests (`task/{formulation,timing_ref,loader_digest}.json`).
- `formulation/prep/` — Public asset fetcher (`fetch_assets.sh`), formulation driver (`formulate.sh`), and 6-test acceptance verifier (`accept.py`).
- `results/` — Official `--final` certificates (`final_check.json`, `final_check_n1.json`, `final_check_default_off.json`), `acceptance_summary.json`, `e2e_summary.json`, `ablation_summary.json`, and per-round `profile.md` / `kernels.md` traces.
- `WORK_LOG.md` — Full engineering log including failed/regression attempts and root-cause diagnoses.
- `docs/UPSTREAM_README.md` — Original upstream `facebookresearch/vggt` README.

---

## Reproduction

```bash
# 1. Fetch public assets (facebook/VGGT-1B model.pt + Virtual KITTI 2.0.3)
export ASSETS=/path/to/assets
bash formulation/prep/fetch_assets.sh

# 2. Run optimized VGGT training directly with upstream entrypoint (headline 2.19x config)
export VGGT_OPT_PIN_MEM=1 VGGT_OPT_GPU_NORM=1 VGGT_OPT_NO_SYNC=1 VGGT_OPT_ROPE_CACHE=1 \
       VGGT_OPT_AGG_BF16_CACHE=1 VGGT_OPT_TRITON_ROPE=1 VGGT_OPT_TRITON_ROPE_NH=1 \
       VGGT_OPT_TRITON_NORM=1 VGGT_OPT_TRITON_FUSED_BLK=1 VGGT_OPT_FAST_QUANTILE=1 \
       VGGT_OPT_MASKED_LOSS=1 VGGT_OPT_DPT_POS_CACHE=1 VGGT_OPT_DPT_CHANNELS_LAST=1 \
       VGGT_OPT_DPT_CHUNK=0 VGGT_OPT_FUSED_ADAMW=1 VGGT_OPT_TRITON_UPSAMPLE=1 \
       VGGT_OPT_HEAD_BF16=depth
cd training
PYTHONPATH=..:. torchrun --nnodes=1 --nproc-per-node=1 launch.py --config default

# 3. Reproduce the full ResearchBoost-Agent frozen judge formulation & verification
#    (requires cloning unmodified facebookresearch/vggt@a288dd0f into targets/reference)
bash formulation/prep/formulate.sh
python3 bench/bench.py check --tree . --config configs/final.json --out runs/repro_final
python3 bench/bench.py check --tree . --config configs/final_n1.json --out runs/repro_n1
python3 bench/bench.py check --quick --tree . --out runs/repro_default_off
```

## Acknowledgments

- Upstream model & training code: [VGGT (`facebookresearch/vggt`)](https://github.com/facebookresearch/vggt) (`a288dd0f`).
- Optimization methodology, frozen judge design, and `numerics/` verification framework: [ResearchBoost-Agent (`DAGroup-PKU/ResearchBoost-Agent`)](https://github.com/DAGroup-PKU/ResearchBoost-Agent) (`0c1c745c`).
