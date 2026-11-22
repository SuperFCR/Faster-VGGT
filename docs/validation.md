# End-to-End Validation (`Faster-VGGT`)

Real end-to-end training, checkpoint save/reload, and held-out validation executed directly through the unmodified upstream entrypoint `training/launch.py` (`torchrun --nnodes=1 --nproc-per-node=1`) on both the unmodified upstream baseline (`targets/reference`, `facebookresearch/vggt @ a288dd0f`) and the optimized `Faster-VGGT` tree (`configs/final.json`, Tier N2).

Raw machine-readable artifact: [`results/e2e_summary.json`](../results/e2e_summary.json).

---

## Protocol

- **Entrypoint**: Unmodified upstream `training/launch.py` + `training/config/default.yaml` (`facebook/VGGT-1B/model.pt`, frozen 908M aggregator + trainable 202M `camera_head` & 12M `depth_head`, `max_img_per_gpu=48`, `accum_steps=2`, `img_size=518`, `AdamW(lr=5e-5, weight_decay=0.05)`).
- **Seed & Schedule**: Identical seed `42`, `max_epochs=1`, `limit_train_batches=40` (`41` logged training steps / `82` microbatches), `save_freq=1` (`checkpoint_0.pt`), `val_epoch_freq=1` on held-out Virtual KITTI 2.0.3 `split=test` (`val_len_test=100`, `10` logged validation batches).
- **Checkpoint Integrity Check**: After each run completes, `checkpoint_0.pt` (`6.17 GiB`, `1,341` keys) is loaded from disk, checked for `0` non-finite tensors, and reloaded with `strict=True` into a fresh `VGGT()` instance.

---

## Held-Out Validation Loss (`VKitti split=test`)

Average across all 10 held-out validation batches after the training epoch (`val_final.*.avg`):

| Arm | `val_loss_camera` | `val_loss_T` | `val_loss_R` | `val_loss_FL` | `val_loss_conf_depth` | `val_loss_reg_depth` | `val_loss_grad_depth` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original (`targets/reference`) | 0.0039 | 0.0026 | 0.0002 | 0.0021 | -0.0542 | 0.0591 | 0.0464 |
| Optimized (`configs/final.json`) | **0.0039** | **0.0027** | **0.0002** | **0.0021** | **-0.0560** | **0.0590** | **0.0466** |
| Absolute Delta (`\|opt - ref\|`) | `0.0000` | `0.0001` | `0.0000` | `0.0000` | `0.0018` | `0.0001` | `0.0002` |

Last validation batch (`val_final.*.last`):

| Arm | `val_loss_camera` | `val_loss_T` | `val_loss_R` | `val_loss_FL` | `val_loss_conf_depth` | `val_loss_reg_depth` | `val_loss_grad_depth` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original (`targets/reference`) | 0.0050 | 0.0037 | 0.0002 | 0.0021 | -0.0579 | 0.0542 | 0.0221 |
| Optimized (`configs/final.json`) | **0.0051** | **0.0039** | **0.0002** | **0.0021** | **-0.0603** | **0.0543** | **0.0223** |

---

## Training Loss & Gradient Norms (41 Steps / 82 Microbatches)

Epoch average (`train_final.*.avg`):

| Arm | `objective` | `loss_camera` | `loss_T` | `loss_R` | `loss_FL` | `loss_conf_depth` | `loss_reg_depth` | `loss_grad_depth` | `Grad/depth` | `Grad/camera` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original (`targets/reference`) | 0.0344 | 0.0049 | 0.0036 | 0.0004 | 0.0017 | -0.0589 | 0.0494 | 0.0537 | 37.8173 | 1.2473 |
| Optimized (`configs/final.json`) | **0.0344** | **0.0049** | **0.0036** | **0.0004** | **0.0018** | **-0.0590** | **0.0493** | **0.0539** | **37.9549** | **1.2713** |

---

## Throughput, Wall Clock, Memory & Checkpoint Integrity

| Metric | Original (`targets/reference`) | Optimized (`configs/final.json`) | Ratio / Delta |
|---|---:|---:|---:|
| Steady-state train `Batch Time` (step 40) | 2.8256 s | **1.2702 s** | **2.22×** |
| Train epoch avg `Batch Time` (incl. cold Triton JIT) | 2.2695 s | **1.3226 s** | **1.72×** |
| Steady-state val `Batch Time` (last batch) | 1.9073 s | **1.0178 s** | **1.87×** |
| Full E2E wall clock (startup + 5 GiB ckpt load + cold JIT + train + 6.17 GiB save + val) | 192.74 s | **145.01 s** | **1.33×** |
| Peak GPU Memory — train (`last` / `avg`) | 29.0 / 27.68 GB | **20.0 / 19.34 GB** | **−9.0 / −8.34 GB** |
| Peak GPU Memory — val (`last` / `avg`) | 32.0 / 32.0 GB | **19.0 / 19.0 GB** | **−13.0 GB** |
| Saved `checkpoint_0.pt` (`6.17 GiB`, `1,341` keys) | `0` non-finite, `strict=True` OK | **`0` non-finite, `strict=True` OK** | Verified |
