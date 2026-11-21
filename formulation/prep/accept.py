"""Pre-optimization judge acceptance suite (ResearchBoost-Agent FORMULATOR_BRIEF.md section 3).

Verifies all mandatory gates before optimization begins:
  - Reference bitwise self-repeatability (2 independent recordings -> 100% bitwise identical)
  - (a) unmodified candidate -> check --quick PASS (bitwise N0), full check UNAVAILABLE (no_benefit)
  - (b) modified entry point / setup AST -> ERROR (Gate 0)
  - (c) semantic mutation (drop depth loss / zero camera loss) -> INCORRECT (Gate 2)
  - (d) mathematically equivalent variant -> PASS on Gate 3 (1-seed 3-sigma)
  - (e) control arm (lr * 1.02) -> INCORRECT on Gate 3 (1-seed 3-sigma)
  - numerics selftest -> ok == True
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "bench" / "bench.py"


def run_cmd(cmd, cwd=ROOT):
    print(f"[accept] running: {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    if r.stderr:
        print(r.stderr[-1500:], file=sys.stderr, flush=True)
    return r


def main():
    py = sys.executable
    out_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (ROOT / "runs" / "acceptance")
    out_root.mkdir(parents=True, exist_ok=True)
    ref_tree = ROOT / "targets" / "reference"
    results = {}

    with tempfile.TemporaryDirectory(prefix="vggt-acc-", dir="/tmp") as tmp:
        tmp = Path(tmp)

        # (a) Unmodified copy of reference: --quick must PASS bitwise N0, full check must be UNAVAILABLE (no_benefit)
        copy_a = tmp / "unmodified"
        shutil.copytree(ref_tree, copy_a, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        run_cmd([py, str(BENCH), "check", "--tree", str(copy_a), "--out", str(out_root / "a_quick"), "--quick"])
        qa = json.loads((out_root / "a_quick" / "check.json").read_text())
        results["a_quick_bitwise"] = {
            "ok": qa["status"] == "PASS" and qa["correctness"].get("bitwise") is True and qa["correctness"].get("measured_tier") == "N0",
            "status": qa["status"],
            "bitwise": qa["correctness"].get("bitwise"),
            "tier": qa["correctness"].get("measured_tier"),
        }

        run_cmd([py, str(BENCH), "check", "--tree", str(copy_a), "--out", str(out_root / "a_full")])
        fa = json.loads((out_root / "a_full" / "check.json").read_text())
        results["a_full_no_benefit"] = {
            "ok": fa["status"] == "UNAVAILABLE" and fa["correctness"].get("verdict") == "no_benefit",
            "status": fa["status"],
            "verdict": fa["correctness"].get("verdict"),
            "measured_speedup": (fa.get("performance") or {}).get("measured_speedup"),
        }

        # (b) Modified entry point AST -> ERROR at Gate 0
        copy_b = tmp / "bad_setup"
        shutil.copytree(copy_a, copy_b)
        lp = copy_b / "training" / "launch.py"
        lp.write_text(lp.read_text() + "\n# setup mutation\ndef _mut(): pass\n")
        # Also mutate main() AST directly
        lp.write_text(lp.read_text().replace("trainer.run()", "x_mut = 1; trainer.run()"))
        run_cmd([py, str(BENCH), "check", "--tree", str(copy_b), "--out", str(out_root / "b_bad_setup"), "--quick"])
        fb = json.loads((out_root / "b_bad_setup" / "check.json").read_text())
        results["b_bad_setup_error"] = {
            "ok": fb["status"] == "ERROR" and fb["correctness"].get("stage") == "setup_guard",
            "status": fb["status"],
            "stage": fb["correctness"].get("stage"),
        }

        # (c) Semantic mutation: zero depth loss gradient (without crashing DDP) -> INCORRECT at Gate 2 (dead_gradient)
        copy_c = tmp / "bad_semantics"
        shutil.copytree(copy_a, copy_c)
        loss_py = copy_c / "training" / "loss.py"
        src = loss_py.read_text()
        assert "total_loss = total_loss + depth_loss" in src
        loss_py.write_text(src.replace("total_loss = total_loss + depth_loss", "total_loss = total_loss + depth_loss * 0.0"))
        run_cmd([py, str(BENCH), "check", "--tree", str(copy_c), "--out", str(out_root / "c_bad_semantics"), "--quick"])
        fc = json.loads((out_root / "c_bad_semantics" / "check.json").read_text())
        results["c_bad_semantics_incorrect"] = {
            "ok": fc["status"] == "INCORRECT",
            "status": fc["status"],
            "stage": fc["correctness"].get("stage"),
            "verdict": fc["correctness"].get("verdict"),
        }

        # (d) Mathematically equivalent variant (TF32 off or reordered ops via AdamW foreach=False): must PASS Gate 3
        copy_d = tmp / "equiv_variant"
        shutil.copytree(copy_a, copy_d)
        opt_py = copy_d / "training" / "train_utils" / "optimizer.py"
        opt_src = opt_py.read_text()
        assert "optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)" in opt_src
        opt_py.write_text(
            opt_src.replace(
                "optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)",
                "optimizer = hydra.utils.instantiate(optimizer_conf, param_groups, foreach=False)",
            )
        )
        run_cmd([py, str(BENCH), "check", "--tree", str(copy_d), "--out", str(out_root / "d_equiv"), "--n1"])
        fd = json.loads((out_root / "d_equiv" / "check.json").read_text())
        results["d_equiv_pass"] = {
            "ok": fd["status"] == "PASS",
            "status": fd["status"],
            "stage": fd["correctness"].get("stage"),
            "verdict": fd["correctness"].get("verdict"),
            "endpoints": ((fd["correctness"].get("trajectory") or {}).get("endpoints")),
        }

        # (e) Control arm: lr * 1.02 -> must FAIL as INCORRECT at Gate 3 (1-seed 3-sigma)
        copy_e = tmp / "lr_control"
        shutil.copytree(copy_a, copy_e)
        opt_py_e = copy_e / "training" / "train_utils" / "optimizer.py"
        opt_src_e = opt_py_e.read_text()
        assert "param_group[option] = scheduler(where)" in opt_src_e
        opt_py_e.write_text(
            opt_src_e.replace(
                "param_group[option] = scheduler(where)",
                "param_group[option] = scheduler(where) * (1.02 if option == 'lr' else 1.0)",
            )
        )
        run_cmd([py, str(BENCH), "check", "--tree", str(copy_e), "--out", str(out_root / "e_lr_control"), "--n1"])
        fe = json.loads((out_root / "e_lr_control" / "check.json").read_text())
        results["e_control_arm_incorrect"] = {
            "ok": fe["status"] == "INCORRECT",
            "status": fe["status"],
            "verdict": fe["correctness"].get("verdict"),
            "endpoints": ((fe["correctness"].get("trajectory") or {}).get("endpoints")),
        }

        # (f) numerics selftest
        st_out = out_root / "numerics_selftest.json"
        r_st = run_cmd([
            py, "-m", "numerics", "selftest",
            "--ref", str(ROOT / "bench" / "records" / "ref"),
            "--variant", str(ROOT / "runs" / "ref_repeat"),
            "--thresholds", str(ROOT / "bench" / "thresholds.json"),
            "--out", str(st_out),
        ])
        st = json.loads(st_out.read_text()) if st_out.exists() else {"ok": False}
        results["f_numerics_selftest"] = {
            "ok": bool(st.get("ok")) and r_st.returncode == 0,
            "mutations": len(st.get("results", [])),
            "failed": [x["mutation"] for x in st.get("results", []) if not x["ok"]],
        }

    summary = {"ok": all(v["ok"] for v in results.values()), "results": results}
    (out_root / "acceptance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
