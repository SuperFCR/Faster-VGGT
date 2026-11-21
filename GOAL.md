# Goal

Bring the **time per training iteration** of VGGT (`facebookresearch/vggt`, commit `a288dd0f14786c93483e45524328726ab7b1b4ce`, `training/config/default.yaml` fine-tuning recipe on Virtual KITTI 2.0.3 resuming from official `facebook/VGGT-1B`) on 1 × NVIDIA A100-SXM4-80GB **down to 1/2 of the reference (a 2.0× speedup)**, with a minimum of **1.3×**, while passing all four gates of the frozen judge below.

- **Training code**: `vggt/`, upstream `facebookresearch/vggt` at commit `a288dd0f14786c93483e45524328726ab7b1b4ce`, branch `opt`; this is where all optimizations live, each behind its own environment-variable switch (off by default).
  - Entry point: `training/launch.py` (`args = parser.parse_args(); cfg = compose(...); trainer = Trainer(**cfg); trainer.run()`), launched with `torchrun --nnodes=1 --nproc-per-node=1` from `training/`.
- **Frozen workload** (`bench/recipe.py`):
  - Exact upstream `training/config/default.yaml` + `default_dataset.yaml` per-GPU load:
    - `model`: `VGGT(img_size=518, patch_size=14, embed_dim=1024, enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)` (~1.12B parameters total: ~908M frozen `aggregator` [DINOv2 ViT-L/14 + 24 frame + 24 global alternating-attention blocks], ~202M trainable `camera_head`, ~12M trainable `depth_head`).
    - Resumes from official public `facebook/VGGT-1B/model.pt` (`strict=False`), `frozen_module_names: ["*aggregator*"]`.
    - Dynamic multi-view batching (`DynamicTorchDataset` + `DynamicBatchSampler` over public Virtual KITTI 2.0.3 via upstream `VKittiDataset`): `img_nums: [2, 24]`, `aspects: [0.33, 1.0]`, `max_img_per_gpu: 48`, `accum_steps: 2`, `num_workers: 8`, `seed_value: 42`, `limit_train_batches: 800`, `max_epochs: 20`.
    - Loss: `MultitaskLoss` (`camera` weight 5.0, L1 over 4 iterative pose stages with `gamma=0.6`; `depth` weight 1.0, confidence-weighted L2 + multi-scale `grad` loss with `valid_range=0.98` quantile outlier filtering).
    - Optimizer: `torch.optim.AdamW(lr=5e-5, weight_decay=0.05)`, `CompositeParamScheduler` (5% linear warmup `1e-8 -> 5e-5`, 95% cosine decay), per-module-group `GradientClipper` (`max_norm=1.0`), `GradScaler(enabled=True)`, `amp_dtype=bfloat16`, `allow_tf32=True`, `cudnn_benchmark=False`.
    - Precision layout as upstream: fp32 master weights and fp32 optimizer state, bf16 autocast inside `aggregator`, `autocast(enabled=False)` (fp32) inside `camera_head`, `depth_head`, and `_process_batch`.
- **Reference** (`targets/reference`, upstream commit unchanged):
  - Frozen timing across three independent cold-cache runs: **`1.9218 s/iter`** (`[57.667, 57.636, 57.656] s` over 30 timed iterations in two 15-iteration windows after 6 warmup iterations), spread **`0.05%`**, peak allocated memory **`28.04 GiB`**, `rng_identical_across_runs = True`.
  - Baseline profile and kernel utilization breakdown are in `task/baseline.md`.

## Single criterion

```bash
python3 bench/bench.py check --tree vggt --config <switch.json> --out <output_dir>
```

Writes `<output_dir>/check.json`.

- **Done**: `status == "PASS"` and `performance.speedup >= 2.0` (with `--final` harness confirmation).
- If profile and kernel-utilization evidence shows 2.0× cannot be reached within the rules, `status == "PASS"` with `performance.speedup >= 1.3` is acceptable for wrap-up once every large `short`, `under-filled`, or `bw-poor` kernel class and repeated chain has been attempted or has a logged reason.

### Gates (evaluated in order; stops at first failure)

0. **Structure** (`coverage`):
   - AST of `training/launch.py` (`main`, `__main__`) and `training/trainer.py` (`Trainer.run`, `Trainer.run_train`, and `Trainer.train_epoch` outside the `for data_iter, batch in enumerate(train_loader):` loop body) must match reference identically.
   - `ComposedDataset` / `VKittiDataset` / `DynamicBatchSampler` must yield bitwise-identical items and first batches on the frozen VKitti fixture (`task/loader_digest.json`).
1. **Benefit**:
   - Agent `check`: one cold-cache run against the frozen reference timing (`task/timing_ref.json`); speedup must exceed `max(1.02, 1 + 2 * frozen_spread)`.
   - Harness `check --final` (requires `task/final-evaluation`): `ref_a -> cand -> ref_b` in three independent fresh processes; identical fingerprints, zero compiles in timing windows, identical per-iteration RNG streams, window bands respected, and speedup `> max(1.02, 1 + 2 * reference_spread)`.
2. **Bitwise** (`numerics` on 2-step deterministic case `train_2step`):
   - Strict deterministic mode (`torch.use_deterministic_algorithms(True, warn_only=False)`, `cudnn.deterministic=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; FlashAttention forward is deterministic for the frozen aggregator so the judged attention path matches the timed path).
   - Every exact check must pass (phase, RNG fingerprints before/after, forward & optimizer call counts, invariants including batch digests/shapes/lr/step/scaler, finiteness, missing tensors, and `dead_gradient`).
   - All 3,776 recorded tensors bitwise identical with unchanged dtypes and math modes → **PASS (N0)**.
3. **Statistical** (only when Gate 2 is not bitwise identical):
   - One paired 20-step deterministic trajectory on the first frozen baseline seed (`seed=42`, `max_epochs=1`, `limit_train_batches=25`), evaluated against 7 endpoints (`objective`, `loss_camera`, `loss_conf_depth`, `loss_reg_depth`, `loss_grad_depth` over steps `[10, 20)`; `step_delta_top1_camera`, `step_delta_top1_depth` over steps `[1, 2)`) using `numerics traj-compare --sigma-k 3` against `bench/traj/frozen.json` (frozen from 6 independent baseline seeds 42–47).
   - Every endpoint `|d| <= 3*sigma` → **PASS** (measured tier `N1` or `N2`, capped at `agent_max_tier = "N2"`). Any endpoint `> 3*sigma` → **INCORRECT**.

## Per-round discipline

1. **Profile first, then change**: run `bench/bench.py profile --tree vggt --config <best.json> --out runs/<round>/profile --iters 5`, review `profile.md` and `kernels.md` (kernel utilization table + repeated chains + bytes model).
2. **One hotspot / bottleneck class per round**: every change sits behind an env switch (off by default, JSON `{"environment": {"VGGT_OPT_...": "1"}}`).
3. **Time only through `bench.py`**: screen with `bench.py time` and `bench.py check --quick`; run full `bench.py check` once gains accumulate.
4. **Log every round in `WORK_LOG.md`** (profile findings, top kernel classes/chains, change, `time` result, `check` result, failed/no-gain rounds included) and commit each accepted change separately on branch `opt`.

## Constraints

- **Unchanged semantics**: same data and order, per-GPU dynamic batching, model architecture, hyperparameters, LR schedule, AdamW update, loss definitions and weights, gradient clipping, and per-step RNG stream. Caching frozen aggregator outputs across steps is forbidden (N5).
- **Precision floor**: upstream precision (`fp32` master weights & optimizer state, `bf16` autocast in aggregator, `fp32` heads, `TF32=True`). No FP8/INT8/FP16 downgrade; local precision changes at most N2 and must pass Gate 3.
- **Same code path**: do not read `VGGT_BENCH_*` variables or behave differently during timing vs judging. If a custom kernel lacks deterministic backward, take an equivalent-precision deterministic path only when `torch.are_deterministic_algorithms_enabled()` is true.
- **Frozen judge**: `bench/`, `numerics/`, `task/`, `targets/` are immutable (`evaluator_sha256` must remain `c5b59bf55f251eca2f3571e0ecba04e39fe9ba32f63fd8fd945e0d1f182c966d`).
