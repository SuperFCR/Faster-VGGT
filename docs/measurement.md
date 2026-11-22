# Measurement & Frozen Judge Protocol (`Faster-VGGT`)

All speedup and correctness claims are certified by the frozen ResearchBoost-Agent judge (`bench/`, `numerics/`, `task/`), locked before optimization began and never modified thereafter.

- **Frozen `evaluator_sha256`**: `c5b59bf55f251eca2f3571e0ecba04e39fe9ba32f63fd8fd945e0d1f182c966d`
- **Certified `source_digest`**: `09c8e56856419bb685f14bc1f0a7bc46416eadcc89bfc27b60949f51d2e71dfe`
- **Hardware**: `1 × NVIDIA A100-SXM4-80GB` (80 GiB HBM2e, 108 SMs), 12 vCPU, Python `3.13.16`, PyTorch `2.14.1+cu130`, Triton `3.8.0`, CUDA `13.0`
- **Raw certificates**: [`results/final_check.json`](../results/final_check.json), [`results/final_check_n1.json`](../results/final_check_n1.json), [`results/final_check_default_off.json`](../results/final_check_default_off.json), [`results/acceptance_summary.json`](../results/acceptance_summary.json)

---

## 1. Pre-Optimization Judge Acceptance Suite (`6/6 PASS`)

Before any optimization was written, the frozen judge passed all 6 negative & control-arm checks ([`formulation/prep/accept.py`](../formulation/prep/accept.py)):

| Check | Purpose | Outcome |
|---|---|---|
| `a_quick_bitwise` | Reference self-repeatability (2-step deterministic) | **`PASS` (`N0`, `3,776/3,776` bitwise)** |
| `a_full_no_benefit` | Reference vs. reference timing must fail benefit gate | **`UNAVAILABLE` (`no_benefit`, `0.9974×`)** |
| `b_bad_setup_error` | Missing asset / broken setup caught before judging | **`ERROR` (`setup_guard`)** |
| `c_bad_semantics_incorrect` | Altered trainer loop structure rejected by AST gate | **`INCORRECT` (`structure`)** |
| `d_equiv_pass` | Exact-math equivalent reorder (`n1`) passes Gate 3 | **`PASS` (`equivalent`, max `0.025σ`)** |
| `e_control_arm_incorrect` | `lr × 1.02` (2% LR perturbation) rejected by Gate 3 | **`INCORRECT` (`+12.18σ` camera / `+9.24σ` depth)** |
| `f_numerics_selftest` | Unmodified `numerics` framework mutation suite | **`12 / 12` mutations caught** |

---

## 2. Steady-State A-B-A Timing (`check --final`)

Each `--final` check executes fresh cold-cache `torchrun --nproc-per-node=1` processes in `ref_a → cand → ref_b` order with `6` warmup iterations followed by `2 × 15 = 30` timed iterations, verifying cold caches, identical math flags (`matmul.allow_tf32=true`, `cudnn.allow_tf32=true`, `cudnn.benchmark=false`), identical data-loader RNG stream, and interference bands (`frozen_band ≤ 5%`, `pair_band ≤ 3%`).

| Config | Tier | `ref_a` (s/iter) | `cand` (s/iter) | `ref_b` (s/iter) | `ref_mean` (s/iter) | `ref_spread` | Speedup | Peak Alloc (GiB) | Peak Reserved (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline (`task/timing_ref.json`, 3 runs) | N0 | — | — | — | 1.9218 | 0.05% | 1.0000× | 28.04 | 32.68 |
| **Headline (`configs/final.json`)** | **N2** | 1.9238 | **0.8807** | 1.9262 | 1.9250 | 0.13% | **2.1857×** | **19.15 (−31.7%)** | **21.53 (−34.1%)** |
| **Pure Lossless (`configs/final_n1.json`)** | **N1** | 1.9252 | **1.0351** | 1.9314 | 1.9283 | 0.32% | **1.8630×** | 30.48 | 35.12 |

---

## 3. Correctness Gates (Gate 2 Bitwise + Gate 3 Trajectory `3σ`)

- **Default-off parity (`results/final_check_default_off.json`)**: Running the modified tree with no `VGGT_OPT_*` flags set passes Gate 2 at **`N0` with `3,776 / 3,776` tensors bitwise identical**.
- **Gate 2 on optimized configs**: `0` errors, `0` exact failures, `0` dtype changes, `0` precision-mode changes (`2,703/3,776` bitwise on N2; `2,704/3,776` bitwise on N1). Non-bitwise tensors arise from fused Triton reductions/orderings (N1) and local BF16 `DPTHead` convolutions (N2), advancing to Gate 3.
- **Gate 3 (20-step deterministic trajectory, `seed=42` vs. 6 frozen baseline seeds `42–47`, bound `±3.0σ`)**:

| Endpoint | Frozen `σ_seed` | Bound (`3σ`) | Headline N2 (`final.json`) `\|d\|/σ` | Pure N1 (`final_n1.json`) `\|d\|/σ` | Control Arm (`lr × 1.02`) `\|d\|/σ` | State |
|---|---:|---:|---:|---:|---:|---|
| `objective` (`[10, 20)`) | `3.245e-2` | `9.734e-2` | `+0.0214σ` | `+0.0158σ` | `+0.0288σ` | within |
| `loss_camera` (`[10, 20)`) | `5.891e-4` | `1.767e-3` | `-0.0642σ` | `-0.0642σ` | `+0.1152σ` | within |
| `loss_conf_depth` (`[10, 20)`) | `2.478e-2` | `7.434e-2` | `+0.0267σ` | `+0.0254σ` | `+0.0199σ` | within |
| `loss_reg_depth` (`[10, 20)`) | `4.662e-3` | `1.399e-2` | `+0.0110σ` | `+0.0126σ` | `+0.0133σ` | within |
| `loss_grad_depth` (`[10, 20)`) | `6.107e-3` | `1.832e-2` | `+0.0282σ` | `+0.0024σ` | `+0.0068σ` | within |
| `step_delta_top1_camera` (`[1, 2)`) | `6.544e-8` | `1.963e-7` | `-0.0521σ` | `-0.0521σ` | **`+12.178σ` (REJECT)** | within |
| `step_delta_top1_depth` (`[1, 2)`) | `8.593e-8` | `2.578e-7` | `+0.0403σ` | `-0.0072σ` | **`+9.239σ` (REJECT)** | within |

Largest endpoint deviation across both certified configs: **`0.0642σ`** (46× below the `3.0σ` threshold).
