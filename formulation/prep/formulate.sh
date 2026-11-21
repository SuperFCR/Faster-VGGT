#!/usr/bin/env bash
set -euo pipefail

cd /content/work/vggt/ws
export ASSETS=/content/assets

echo "=== F1: Loader digest ==="
python3 bench/bench.py loader --tree targets/reference > task/loader_digest.json

echo "=== F2: Baseline 3-run timing ==="
rm -rf runs/baseline
python3 bench/bench.py baseline --out runs/baseline
cp runs/baseline/baseline.json task/timing_ref.json
python3 -c '
import json
d = json.loads(open("task/timing_ref.json").read())
print("BASELINE TIMING:", {k: d.get(k) for k in ("status", "total_s_runs", "total_s", "s_per_iteration", "spread", "window_ratio", "peak_memory_allocated_gib", "rng_identical_across_runs")})
assert d["status"] == "ok" and d["rng_identical_across_runs"] is True
'

echo "=== F3: Baseline profile (5 iters + kernel_util) ==="
rm -rf runs/baseline_profile
python3 bench/bench.py profile --tree targets/reference --out runs/baseline_profile --iters 5

echo "=== F4: Deterministic 2-step bitwise records (2x ref + 1x variant for calibration) ==="
rm -rf bench/records/ref runs/ref_repeat runs/var_tf32_off
python3 bench/bench.py record --tree targets/reference --out bench/records/ref
python3 bench/bench.py record --tree targets/reference --out runs/ref_repeat
python3 bench/bench.py record --tree targets/reference --out runs/var_tf32_off --variant tf32_off
python3 -m numerics calibrate --ref bench/records/ref --variants runs/ref_repeat runs/var_tf32_off --out bench/thresholds.json
python3 -m numerics compare --ref bench/records/ref --cand runs/ref_repeat --thresholds bench/thresholds.json --out runs/ref_self_compare.json
python3 -c '
import json
c = json.loads(open("runs/ref_self_compare.json").read())
print("SELF COMPARE:", {k: c.get(k) for k in ("status", "measured_tier", "checked", "bitwise_identical", "bitwise_trusted")})
assert c["status"] == "PASS" and c["checked"] == c["bitwise_identical"] and c["measured_tier"] == "N0"
'

echo "=== F5: 6-seed deterministic reference trajectories -> freeze sigma ==="
rm -rf bench/traj/ref
python3 bench/bench.py record --tree targets/reference --out bench/traj/ref --traj-seeds 42,43,44,45,46,47
python3 -m numerics traj-freeze --ref bench/traj/ref --spec bench/traj/spec.json --k 2.0 --out bench/traj/frozen.json
cat bench/traj/frozen.json

echo "=== F6: Freeze input fingerprints & evaluator_sha256 ==="
python3 bench/bench.py freeze-inputs \
  /content/assets/VGGT-1B/model.pt \
  /content/assets/vkitti/sequence_list.txt \
  /content/assets/vkitti/Scene01/clone/extrinsic.txt \
  /content/assets/vkitti/Scene01/clone/intrinsic.txt \
  /content/work/vggt/ws/task/loader_digest.json
cat task/formulation.json

echo "=== F7: Pre-optimization acceptance suite ==="
python3 formulation/prep/accept.py runs/acceptance
