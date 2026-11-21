"""VGGT training optimization bench (ResearchBoost-Agent frozen judge, current lite-check rules).

Commands:
  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick] [--skip-timing] [--n1]
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--nsys] [--iters N]
  python bench/bench.py baseline --out <dir>
  python bench/bench.py record   --tree <dir> --out <dir> [--cases <file>] [--variant <v>] [--traj-seeds 1,2,...]
  python bench/bench.py loader   --tree <dir>
  python bench/bench.py freeze-inputs <path> ...

Verdict gates (docs/judge.md & FORMULATOR_BRIEF.md section 2):
  0. Structure: AST comparison of training/launch.py (`main`, `__main__`) and training/trainer.py (`run`,
     `run_train`, `train_epoch` outside the per-step `for data_iter, batch in enumerate(train_loader):` loop
     body), plus bitwise-identical dataset items and first batches on the frozen VKitti fixture.
  1. Benefit: fresh torchrun processes of training/launch.py with cold compile caches.
     - Agent `check`: candidate timed once against the frozen reference timing (task/timing_ref.json);
       speedup must exceed max(1.02, 1 + 2 * frozen_spread).
     - Harness `check --final` (requires task/final-evaluation marker): reference_a -> candidate -> reference_b
       with interference checks (frozen_band, pair_band, window_band, window_band_candidate);
       speedup must exceed max(1.02, 1 + 2 * reference_spread).
     - Rejected as UNAVAILABLE (no_benefit) if below bar; precision is not evaluated.
  2. Bitwise: 2-step deterministic case via `numerics` (exact items first -> INCORRECT; missing tensors ->
     exact failure INCORRECT; all tensors bitwise identical + no dtype/math-mode change -> PASS N0).
  3. Statistical: only when not bitwise identical. One paired deterministic trajectory on the first frozen
     baseline seed (`numerics traj-compare --sigma-k 3`), where sigma is frozen from 6 independent baseline
     seeds before any candidate is seen. All endpoints |d| <= 3*sigma -> PASS (N1/N2); any beyond -> INCORRECT.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent


def _expand(value):
    if isinstance(value, str) and "$" in value:
        out = os.path.expandvars(value)
        if "$" in out:
            raise SystemExit(f"{value}: set ASSETS (and any other variable it names) before running the bench")
        return out
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


CFG = _expand(json.loads((BENCH / "bench.json").read_text()))
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402

LOCK = Path(CFG["gpu_lock"])
PY = Path(CFG["python"])
TORCHRUN = Path(CFG["torchrun"]) if CFG.get("torchrun") else Path(shutil.which("torchrun") or str(PY.parent / "torchrun"))
TIERS = ["N0", "N1", "N2", "N3", "N4", "N5"]
SCRATCH = Path(CFG.get("scratch", "/tmp"))
TIMEOUT = CFG["timeouts_s"]
RESERVED = set(CFG["reserved_env"])
SIGNATURES = [(re.compile(x["regex"]), x["reason"]) for x in CFG.get("invalid_run_signatures", [])]
COST = {"gpu_seconds": 0.0, "gpu_runs": 0}
TREE_SKIP = {".git", "__pycache__", "wandb", "outputs", "figures", "logs", "examples", "docs"}


def path(key):
    p = Path(_expand(key))
    return p if p.is_absolute() else (ROOT / p).resolve()


def log(msg):
    print(f"[bench {datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    """Joint SHA-256 of the frozen judge: numerics/, bench/ code & config & frozen manifests."""
    files = sorted(p for p in (ROOT / "numerics").rglob("*.py"))
    files += sorted(
        p
        for p in BENCH.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in (".py", ".json", ".yaml")
        and "ref" not in p.parts
    )
    if (BENCH / "records").exists():
        files += sorted(p for p in (BENCH / "records").rglob("*.json"))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------- GPU runs

class GpuLock:
    def __init__(self, what, timeout=8 * 3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(LOCK, "a+")
        t0, said = time.time(), False
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not said:
                    log(f"waiting for {LOCK} ({self.what}); do not wrap bench.py in flock")
                    said = True
                if time.time() - t0 > self.timeout:
                    raise SystemExit(f"could not get {LOCK} within {self.timeout}s for {self.what}")
                time.sleep(5)
        self.t0 = time.time()
        log(f"GPU lock held: {self.what}")
        return self

    def __exit__(self, *_):
        COST["gpu_seconds"] += (time.time() - self.t0) * CFG["world_size"]
        COST["gpu_runs"] += 1
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cost():
    return {
        "gpu_hours": round(COST["gpu_seconds"] / 3600, 4),
        "gpu_runs": COST["gpu_runs"],
        "method": f'GPU lock hold time x {CFG["world_size"]} GPU(s) (startup, warmup and failed runs included)',
    }


def scratch_dir(prefix):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(SCRATCH).free / 2**30
    min_free = CFG.get("min_scratch_free_gib", 15)
    if free < min_free:
        raise SystemExit(f"{SCRATCH} has only {free:.1f} GiB free (< {min_free}); clean up before measuring")
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = []
    for line in q.stdout.strip().splitlines():
        idx, uid, mem, util = [x.strip() for x in line.split(",")]
        out.append({"index": int(idx), "uuid": uid, "memory_mib": int(mem), "util": int(util)})
    return out


def wait_idle(what, patience=300):
    t0 = time.time()
    while True:
        state = gpu_state()
        busy = [g for g in state if g["memory_mib"] > 1024 or g["util"] > 5]
        if not busy:
            return state, None
        if time.time() - t0 > patience:
            return state, f"GPUs busy for {patience}s before {what}: {busy}"
        time.sleep(10)


def _tagged(tag):
    me, pids = os.getuid(), []
    mark = f"VGGT_BENCH_RUN={tag}".encode()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid == me and mark in (d / "environ").read_bytes().split(b"\0"):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _kill_tagged(tag):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _tagged(tag)
        if not pids:
            return 0
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        time.sleep(2)
    return len(_tagged(tag))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def base_env(tmp, cand_env=None, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith("VGGT_BENCH")}
    env.update(cand_env or {})
    caches = Path(tmp) / "caches"
    env.update(
        {
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "TORCHINDUCTOR_CACHE_DIR": str(caches / "inductor"),
            "TRITON_CACHE_DIR": str(caches / "triton"),
            "CUDA_CACHE_PATH": str(caches / "cuda"),
            "XDG_CACHE_HOME": str(caches / "xdg"),
            "MPLCONFIGDIR": str(caches / "mpl"),
            "HF_HOME": CFG["hf_home"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "PYTHONNOUSERSITE": "1",
            "PATH": f'{PY.parent}:{os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}',
            **({"LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"]} if "LD_LIBRARY_PATH" in os.environ else {}),
        }
    )
    env.pop("PYTHONPATH", None)
    if pythonpath:
        env["PYTHONPATH"] = ":".join(str(p) for p in pythonpath)
    for v in ("inductor", "triton", "cuda", "xdg", "mpl"):
        (caches / v).mkdir(parents=True, exist_ok=True)
    return env


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    tag = uuid.uuid4().hex[:12]
    with GpuLock(what):
        before, problem = wait_idle(what)
        if problem:
            return {"status": "INVALID", "reason": problem, "returncode": None, "gpus_before": before}
        env = dict(env, VGGT_BENCH_RUN=tag)
        if CFG.get("cpu_affinity"):
            cmd = ["taskset", "-c", CFG["cpu_affinity"], *cmd]
        t0 = time.time()
        status = None
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc, status = None, "TIMEOUT"
        wall = time.time() - t0
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        left = _kill_tagged(tag)
        after, _ = wait_idle(f"{what} (after)", patience=60)
    res = {
        "status": status or "ok",
        "returncode": rc,
        "wall_s": round(wall, 1),
        "gpus_before": before,
        "gpus_after": after,
    }
    if status == "TIMEOUT":
        res["reason"] = f"{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}"
    if left:
        res["warning"] = f"{left} processes of the run survived SIGKILL"
    if any(g["memory_mib"] > 1024 for g in after or []):
        res["warning"] = (res.get("warning", "") + " GPU memory still in use 60s after the run").strip()
    return res


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind("\n", 0, m.start()) + 1 : text.find("\n", m.end())]
            hits.append({"reason": reason, "line": line.strip()[:300]})
    return hits


def launch(argv, port=None):
    return [
        str(TORCHRUN),
        "--nnodes=1",
        f'--nproc-per-node={CFG["world_size"]}',
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(port or free_port()),
        *argv,
    ]


# ---------------------------------------------------------------- inputs & sources

def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for f in sorted(tree.rglob("*")):
        rel = f.relative_to(tree)
        if any(part in TREE_SKIP for part in rel.parts) or f.is_symlink() or not f.is_file():
            continue
        if rel.name == f"{recipe.CONFIG_NAME}.yaml":
            continue
        h.update(str(rel).encode() + b"\0" + sha256_file(f).encode() + b"\n")
    return h.hexdigest()


def file_fingerprint(p, kind):
    if kind == "sha256":
        return sha256_file(p)
    st = Path(p).stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def verify_inputs(when):
    form_path = ROOT / "task" / "formulation.json"
    if not form_path.exists():
        return {"when": when, "ok": True, "mismatches": {}, "checked": 0, "seconds": 0.0}
    spec = json.loads(form_path.read_text()).get("inputs", {})
    t0, bad = time.time(), {}
    for rel, want in spec.items():
        p = path(rel)
        got = file_fingerprint(p, want["kind"]) if p.exists() else None
        if got != want["value"]:
            bad[rel] = {"want": want["value"], "got": got}
    log(f"inputs {when}: {len(spec) - len(bad)}/{len(spec)} match the formulation ({time.time() - t0:.1f}s)")
    return {"when": when, "ok": not bad, "mismatches": bad, "checked": len(spec), "seconds": round(time.time() - t0, 1)}


def freeze_inputs(paths, big=200 << 20):
    out = {}
    for root in paths:
        root = Path(root)
        files = (
            [root]
            if root.is_file()
            else sorted(f for f in root.rglob("*") if f.is_file() and not f.name.endswith(".lock"))
        )
        for f in files:
            kind = "stat" if f.stat().st_size > big else "sha256"
            out[str(f)] = {"kind": kind, "value": file_fingerprint(f, kind)}
    return out


def check_reference():
    ref = path(CFG["reference"])
    git = ["git", "-c", "safe.directory=*", "-C", str(ref)]
    dirty = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
    ).stdout.strip()
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if dirty or head != CFG["reference_commit"]:
        raise SystemExit(f"reference tree is not the frozen upstream commit (head {head}, dirty: {bool(dirty)})")
    return ref


def materialize(src, dest):
    def ignore(directory, names):
        return [n for n in names if n in TREE_SKIP or n.endswith(".pyc") or n == f"{recipe.CONFIG_NAME}.yaml"]

    shutil.copytree(src, dest, ignore=ignore, symlinks=False)
    for d in [Path(dest), *(p for p in Path(dest).rglob("*") if p.is_dir())]:
        d.chmod(d.stat().st_mode | 0o700)
    return Path(dest)


def candidate_environment(config_file):
    if not config_file:
        return {}
    data = json.loads(Path(config_file).read_text())
    extra = sorted(set(data) - {"environment"})
    if extra:
        raise SystemExit(f'{config_file}: only an "environment" mapping is accepted, found {extra}')
    env = {str(k): str(v) for k, v in (data.get("environment") or {}).items()}
    bad = sorted(k for k in env if k in RESERVED or k.startswith("VGGT_BENCH"))
    if bad:
        raise SystemExit(f"{config_file}: {bad} are managed by the bench and cannot be candidate switches")
    return env


def versions():
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "torchvision", "triton", "hydra-core", "omegaconf", "fvcore", "numpy"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


FINGERPRINT_KEYS = ("argv", "math_mode", "gpus", "compile_cache", "instrumentation", "versions", "timing_plan")


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ["missing fingerprint"]
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 0: structure

def setup_fingerprint(tree):
    """AST of launch.py (main, __main__) and trainer.py (run, run_train, train_epoch outside per-step loop)."""
    parts = {}
    launch_mod = ast.parse((Path(tree) / recipe.ENTRY_LAUNCH).read_text())
    for node in launch_mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            parts["launch.main"] = ast.dump(node, include_attributes=False)
        elif (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            parts["launch.__main__"] = ast.dump(node, include_attributes=False)

    trainer_mod = ast.parse((Path(tree) / recipe.ENTRY_TRAINER).read_text())
    loops = 0
    for node in trainer_mod.body:
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in ("run", "run_train"):
                    parts[f"Trainer.{fn.name}"] = ast.dump(fn, include_attributes=False)
                elif isinstance(fn, ast.FunctionDef) and fn.name == "train_epoch":
                    for inner in ast.walk(fn):
                        if (
                            isinstance(inner, ast.For)
                            and isinstance(inner.target, ast.Tuple)
                            and len(inner.target.elts) == 2
                            and isinstance(inner.target.elts[0], ast.Name)
                            and inner.target.elts[0].id == "data_iter"
                        ):
                            inner.body = [ast.Pass()]
                            loops += 1
                    parts["Trainer.train_epoch"] = ast.dump(fn, include_attributes=False)
    return parts, loops


def setup_guard(ref, tree):
    want, _ = setup_fingerprint(ref)
    try:
        got, loops = setup_fingerprint(tree)
    except (OSError, SyntaxError) as exc:
        return [f"entry unreadable: {exc}"]
    problems = []
    if loops != 1:
        problems.append(f"expected exactly one `for data_iter, batch in ...` loop in Trainer.train_epoch(), found {loops}")
    for name in ("launch.main", "launch.__main__", "Trainer.run", "Trainer.run_train", "Trainer.train_epoch"):
        if got.get(name) != want.get(name):
            problems.append(
                f"{name} differs from the reference"
                + (" outside the per-step loop body" if name == "Trainer.train_epoch" else "")
            )
    return problems


LOADER_PROBE = r'''
import hashlib, json, os, random, sys
from pathlib import Path
import numpy as np
import torch

tree = Path(sys.argv[1]).resolve()
vkitti = sys.argv[2]
os.chdir(tree / "training")
sys.path.insert(0, str(tree))
sys.path.insert(0, str(tree / "training"))

from omegaconf import OmegaConf
from data.composed_dataset import ComposedDataset
from data.dynamic_dataloader import DynamicBatchSampler, DynamicDistributedSampler

random.seed(42); np.random.seed(42); torch.manual_seed(42)
common = OmegaConf.create({
    "fix_img_num": -1, "fix_aspect_ratio": 1.0, "load_track": False, "track_num": 1024,
    "training": True, "inside_random": True, "img_size": 518, "patch_size": 14,
    "rescale": True, "rescale_aug": True, "landscape_check": False, "debug": False,
    "get_nearby": True, "load_depth": True, "img_nums": [2, 24], "max_img_per_gpu": 48,
    "allow_duplicate_img": True, "repeat_batch": False,
    "augs": {"cojitter": True, "cojitter_ratio": 0.3, "scales": [0.8, 1.2], "aspects": [0.33, 1.0],
             "color_jitter": {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.1, "p": 0.9},
             "gray_scale": True, "gau_blur": False},
})
ds_cfg = [{"_target_": "data.datasets.vkitti.VKittiDataset", "split": "train",
           "VKitti_DIR": vkitti, "len_train": 100000, "expand_ratio": 8}]
ds = ComposedDataset(dataset_configs=ds_cfg, common_config=common)

def dig(d):
    h = hashlib.sha256()
    for k in sorted(d):
        v = d[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode() + str(tuple(v.shape)).encode())
            h.update(v.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()

items = []
for idx in [(0, 4, 0.75), (1, 6, 0.5), (2, 8, 1.0)]:
    random.seed(100 + idx[0]); np.random.seed(100 + idx[0]); torch.manual_seed(100 + idx[0])
    items.append(dig(ds[idx]))

random.seed(840); np.random.seed(840); torch.manual_seed(840)
sampler = DynamicDistributedSampler(ds, num_replicas=1, rank=0, seed=42, shuffle=True)
bsampler = DynamicBatchSampler(sampler, [0.33, 1.0], [2, 24], seed=42, max_img_per_gpu=48)
dl = torch.utils.data.DataLoader(ds, num_workers=0, pin_memory=False, batch_sampler=bsampler)
it = iter(dl)
batches = [dig(next(it)) for _ in range(3)]
print(json.dumps({"len": len(ds), "items": items, "batches": batches}))
'''


def loader_digest(tree):
    with scratch_dir("vggt-ld-") as tmp:
        env = base_env(tmp)
        r = subprocess.run(
            [str(PY), "-c", LOADER_PROBE, str(tree), str(path(CFG["fixture"]["vkitti"]))],
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    if r.returncode:
        return None, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), None


def coverage(tree):
    ref_src = check_reference()
    problems = setup_guard(ref_src, tree)
    if problems:
        return "ERROR", {
            "passed": False,
            "stage": "setup_guard",
            "problems": problems,
            "note": "the numerics adapter reproduces this setup from the reference and executes only the per-step "
            "loop body of Trainer.train_epoch(); move changes into Trainer.__init__, called helpers, model modules, "
            "or inside the loop body",
        }
    frozen = json.loads((ROOT / "task" / "loader_digest.json").read_text())
    got, err = loader_digest(tree)
    if got != frozen:
        return "INCORRECT", {
            "passed": False,
            "stage": "loader_check",
            "error": err,
            "note": "dataset classes yield different items or first batches than the reference on the frozen fixture",
        }
    return None


# ---------------------------------------------------------------- gate 2: bitwise records

def adapter_kwargs(tree, scratch, variant=None):
    kw = {
        "repo": str(tree),
        "vkitti": str(path(CFG["fixture"]["vkitti"])),
        "ckpt": str(path(CFG["fixture"]["ckpt"])),
        "scratch": str(scratch),
    }
    if variant:
        kw["variant"] = variant
    return kw


def record_cases(tree, rec, cases, logs_dir, cand_env, label, variant=None, losses_only=False, sketch=True):
    n = CFG["numerics"]
    for case in cases:
        with scratch_dir("vggt-r-") as tmp:
            cases_file = Path(tmp) / "case.json"
            cases_file.write_text(json.dumps([case]))
            env = base_env(tmp, cand_env, pythonpath=[ROOT])
            cmd = launch(
                [
                    "-m",
                    "numerics",
                    "record",
                    "--adapter",
                    n["adapter"],
                    "--adapter-kwargs",
                    json.dumps(adapter_kwargs(tree, Path(tmp) / "train", variant)),
                    "--cases",
                    str(cases_file),
                    *(
                        ["--losses-only"]
                        if losses_only
                        else (["--sketch-dim", str(n["sketch_dim"])] if sketch else [])
                    ),
                    "--math-mode",
                    CFG["math_mode"],
                    "--deterministic",
                    "--out",
                    str(rec),
                ]
            )
            log_file = Path(logs_dir) / f'{label}_{case["name"]}.log'
            r = run_gpu(
                cmd,
                ROOT,
                env,
                log_file,
                f'{label}: case {case["name"]}',
                TIMEOUT["trajectory" if losses_only else "numerics"],
            )
        if r["status"] != "ok" or r["returncode"]:
            return {
                "status": "INVALID" if r["status"] == "INVALID" else "ERROR",
                "reason": r.get("reason") or f'record exited {r["returncode"]}; see {log_file.name}',
            }
    return None


def missing_tensor_failures(report):
    out = []
    for err in report.get("errors") or []:
        if isinstance(err, dict) and err.get("missing") and not err.get("unexpected"):
            out.append(
                {
                    "where": err["where"],
                    "check": "missing_tensors",
                    "note": "the candidate produced none of these tensors (e.g. missing gradient): different work",
                    "missing": err["missing"][:20],
                }
            )
    return out


def dead_gradient_failures(ref_dir, cand_dir):
    """Exact check (docs/judge.md): a gradient that is nonzero in the reference must remain nonzero in the candidate."""
    out = []
    for rp in sorted(Path(ref_dir).glob("*.r*.json")):
        cp = Path(cand_dir) / rp.name
        if not cp.exists():
            continue
        rm = json.loads(rp.read_text())
        cm = json.loads(cp.read_text())
        for rs, cs in zip(rm.get("steps", []), cm.get("steps", [])):
            dead = []
            for name, rd in rs.get("tensors", {}).items():
                if not name.startswith("grad/"):
                    continue
                cd = cs.get("tensors", {}).get(name)
                rn = rd.get("norm") or 0.0
                cn = (cd.get("norm") or 0.0) if cd else 0.0
                if rn > 0.0 and cn == 0.0:
                    dead.append(name)
            if dead:
                out.append(
                    {
                        "where": f'{rm["case"]["name"]} r{rm["rank"]} s{rs["step"]}',
                        "check": "dead_gradient",
                        "note": "gradient is nonzero in reference but identically zero in candidate",
                        "tensors": dead[:20],
                    }
                )
    return out


def numerics(tree, out, cand_env):
    n = CFG["numerics"]
    rec = out / "records"
    shutil.rmtree(rec, ignore_errors=True)
    cases = json.loads(path(n["cases"]).read_text())
    failed = record_cases(tree, rec, cases, out, cand_env, "record")
    if failed:
        return failed
    report = out / "compare.json"
    env = base_env(out, pythonpath=[ROOT])
    r = subprocess.run(
        [
            str(PY),
            "-m",
            "numerics",
            "compare",
            "--ref",
            str(path(n["ref"])),
            "--cand",
            str(rec),
            "--thresholds",
            str(path(n["thresholds"])),
            *[a for x in n.get("exclude", []) for a in ("--exclude", f'{x["regex"]}={x["reason"]}')],
            "--out",
            str(report),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    (out / "compare.log").write_text(r.stdout + r.stderr)
    shutil.rmtree(out / "caches", ignore_errors=True)
    if not report.exists():
        return {"status": "ERROR", "reason": f"compare exited {r.returncode}; see compare.log"}
    data = json.loads(report.read_text())
    dead_g = dead_gradient_failures(path(n["ref"]), rec)
    if not n.get("keep_records"):
        shutil.rmtree(rec, ignore_errors=True)
    data["exact_failures"] = (data.get("exact_failures") or []) + missing_tensor_failures(data) + dead_g
    summary = {
        k: (len(v) if isinstance(v, (list, dict)) else v)
        for k, v in data.items()
        if k not in ("status", "measured_tier")
    }
    bitwise = (
        data.get("checked") == data.get("bitwise_identical")
        and not data.get("dtype_changes")
        and not data.get("precision_mode_changes")
        and data.get("bitwise_trusted", False)
    )
    return {
        "status": data.get("status"),
        "measured_tier": data.get("measured_tier"),
        "report": str(report),
        "errors": len(data.get("errors") or []),
        "exact_failures": data.get("exact_failures") or [],
        "bitwise": bitwise,
        "summary": summary,
    }


# ---------------------------------------------------------------- gate 3: trajectories (current 1-seed 3-sigma rule)

def traj_case(spec, seed):
    return dict(spec["case"], name=f"traj_s{seed}", seed=seed)


def trajectories(tree, out, cand_env, variant=None):
    tj = CFG["trajectory"]
    spec = json.loads(path(tj["spec"]).read_text())
    frozen_path = path(tj["frozen"])
    frozen_seeds = json.loads(frozen_path.read_text())["reference_seeds"]
    n_seeds = int(tj.get("seeds", 1))
    sigma_k = float(tj.get("sigma_k", 3.0))
    seeds = frozen_seeds[:n_seeds]
    rec = out / "trajectory"
    shutil.rmtree(rec, ignore_errors=True)
    failed = record_cases(
        tree, rec, [traj_case(spec, s) for s in seeds], out, cand_env, "traj", variant=variant, losses_only=True
    )
    if failed:
        return {"verdict": "error", **failed}
    report_path = out / f"traj_report_{n_seeds}.json"
    env = base_env(out, pythonpath=[ROOT])
    subprocess.run(
        [
            str(PY),
            "-m",
            "numerics",
            "traj-compare",
            "--frozen",
            str(frozen_path),
            "--cand",
            str(rec),
            "--sigma-k",
            str(sigma_k),
            "--out",
            str(report_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    shutil.rmtree(out / "caches", ignore_errors=True)
    if not report_path.exists():
        return {"verdict": "error", "reason": "traj-compare produced no report"}
    report = json.loads(report_path.read_text())
    report["report"] = str(report_path)
    return report


# ---------------------------------------------------------------- gate 1: timing

def time_tree(tree, out, label, cand_env=None, profile=None, profile_iters=5):
    t = CFG["timing"]
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f"timing_{label}"
    for f in out.glob(f"timing_{label}.r*.json"):
        f.unlink()
    with scratch_dir("vggt-t-") as tmp:
        recipe.write_config(
            tree,
            Path(tmp) / "train",
            path(CFG["fixture"]["vkitti"]),
            path(CFG["fixture"]["ckpt"]),
            config_name=recipe.CONFIG_NAME,
            seed=42,
            max_epochs=20,
            limit_train_batches=800,
            limit_val_batches=400,
            cudnn_deterministic=False,
            num_workers=int(CFG.get("num_workers", 8)),
            save_freq=0,
        )
        train_args = ["--config", recipe.CONFIG_NAME]
        hook_args = [
            "--warmup",
            str(t["warmup"]),
            "--window",
            str(t["window"]),
            "--repeats",
            str(t["repeats"]),
            *(["--profile", profile, "--profile-iters", str(profile_iters)] if profile else []),
        ]
        argv = launch([str(BENCH / "timing.py"), "--out", str(prefix), *hook_args, "--", *train_args])
        stable_argv = [f'torchrun --nproc-per-node={CFG["world_size"]}', "timing.py", *hook_args, "--", *train_args]
        env = base_env(tmp, cand_env)
        if profile == "nsys":
            nsys = shutil.which("nsys") or CFG.get("nsys")
            if not nsys or not Path(nsys).exists():
                raise SystemExit("nsys not found")
            argv = [
                nsys,
                "profile",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--cuda-graph-trace=node",
                "--trace=cuda,nvtx,osrt,cublas,cudnn",
                "--sample=none",
                "--cuda-memory-usage=false",
                "--force-overwrite=true",
                "-o",
                str(out / f"nsys_{label}"),
                *argv,
            ]
        log_file = out / f"timing_{label}.log"
        what = f'timing {label}{f" ({profile})" if profile else ""}'
        r = run_gpu(argv, Path(tree) / "training", env, log_file, what, TIMEOUT["profile" if profile else "timing"])
        # Remove temporary config file written into materialized tree so tree_digest stays unchanged
        tmp_cfg = Path(tree) / "training" / "config" / f"{recipe.CONFIG_NAME}.yaml"
        if tmp_cfg.exists():
            tmp_cfg.unlink()

    plan = [t["warmup"], t["window"], t["repeats"]] if not profile else [t["warmup"], profile_iters, 1]
    fingerprint = {
        "argv": stable_argv,
        "candidate_environment": cand_env or {},
        "math_mode": CFG["math_mode"],
        "gpus": [(g["index"], g["uuid"]) for g in r.get("gpus_before") or []],
        "compile_cache": "cold (fresh per run)",
        "instrumentation": profile or "none",
        "timing_plan": plan,
        "versions": versions(),
    }
    run = {
        "run_status": r["status"],
        "wall_s": r.get("wall_s"),
        "fingerprint": fingerprint,
        "log": str(log_file),
        **({"warning": r["warning"]} if r.get("warning") else {}),
    }
    if r["status"] != "ok":
        return {"status": r["status"], "reason": r["reason"], **run}
    ranks = []
    for rank in range(CFG["world_size"]):
        f = Path(f"{prefix}.r{rank}.json")
        ranks.append(json.loads(f.read_text()) if f.exists() else None)
    if r["returncode"] or any(x is None or x["status"] != "ok" for x in ranks):
        errors = [x["error"] for x in ranks if x and x.get("error")]
        return {
            "status": "ERROR",
            "reason": f'timing exited {r["returncode"]}; {errors[:1]}; see {log_file.name}',
            **run,
        }
    hits = signatures(log_file)
    if hits:
        return {
            "status": "INVALID",
            "reason": "invalid-run signature in the log: " + hits[0]["reason"],
            "signatures": hits,
            **run,
        }
    return {**summarize_timing(ranks, r["wall_s"], profile, profile_iters), **run, "_ranks": ranks}


def summarize_timing(ranks, wall, profile=None, profile_iters=5):
    t = CFG["timing"]
    repeats = 1 if profile else t["repeats"]
    size = profile_iters if profile else t["window"]
    per_window = {}
    for x in ranks:
        for w in x["windows"]:
            prev = per_window.get(w["k"])
            if prev is None or w["seconds"] > prev["seconds"]:
                per_window[w["k"]] = w
    problems = []
    ws = [per_window[k] for k in range(repeats) if k in per_window]
    if len(ws) != repeats:
        problems.append(f"{len(ws)} of {repeats} windows")
    for w in ws:
        if w.get("compiles_in_window"):
            problems.append(
                f'window {w["k"]}: torch.compile compiled {w["compiles_in_window"]} graphs inside the timing window'
            )
    secs = [w["seconds"] for w in ws]
    total = sum(secs) if secs else None
    frozen = None
    ref_file = ROOT / "task" / "timing_ref.json"
    if not profile and ref_file.exists():
        frozen = json.loads(ref_file.read_text())
        n_check = t["warmup"] + t["window"] * t["repeats"]
        for x in ranks:
            want = frozen["rng_fingerprints"][str(x["rank"])][:n_check]
            if x["rng_fingerprints"][:n_check] != want:
                k = next((i for i, (a, b) in enumerate(zip(x["rng_fingerprints"], want)) if a != b), None)
                problems.append(
                    f'rank {x["rank"]}: random-number stream differs from frozen reference at iteration {k}'
                )
                break
    res = {
        "status": "ok" if not problems and secs else "ERROR",
        "problems": problems,
        "window_s": [round(s, 3) for s in secs],
        "total_s": round(total, 3) if total else None,
        "iterations_timed": size * repeats,
        "s_per_iteration": round(total / (size * len(secs)), 4) if secs else None,
        "peak_memory_allocated_gib": round(max(x["peak_memory_allocated_bytes"] for x in ranks) / 2**30, 2),
        "peak_memory_reserved_gib": round(max(x["peak_memory_reserved_bytes"] for x in ranks) / 2**30, 2),
        "startup_to_first_iteration_s": round(max(x["startup_to_first_iteration_s"] or 0 for x in ranks), 1),
        "process_wall_s": round(wall, 1),
        "math_flags_at_end": ranks[0].get("math_flags_at_end"),
    }
    if len(secs) == 2:
        res["window_ratio"] = round(secs[0] / secs[1], 4)
        if frozen and frozen.get("window_ratio"):
            res["window_ratio_vs_frozen"] = round(res["window_ratio"] / frozen["window_ratio"] - 1, 4)
    return res


def strip(run):
    return {k: v for k, v in run.items() if k != "_ranks"} if run else run


def benefit_lite(tree, out, cand_env):
    """Agent check Gate 1: one candidate timing run against the frozen baseline timing."""
    t = CFG["timing"]
    frozen = json.loads((ROOT / "task" / "timing_ref.json").read_text())
    cand = time_tree(tree, out, "cand", cand_env=cand_env)
    runs = {"cand": strip(cand)}
    if cand["status"] != "ok":
        status = "INVALID" if cand["status"] in ("INVALID", "TIMEOUT") else "ERROR"
        return status, {"speedup": None, "runs": runs, "reason": cand.get("reason"), "problems": cand.get("problems")}
    ref_total = frozen["total_s"]
    frozen_spread = frozen.get("spread", 0.0)
    c = cand["total_s"]
    notes = []
    dev = cand.get("window_ratio_vs_frozen")
    if dev is not None and abs(dev) > t["window_band_candidate"]:
        notes.append(f'candidate window ratio {cand["window_ratio"]} is {dev:+.1%} off frozen ratio (note only)')
    bar = max(CFG["benefit"]["min_speedup"], 1 + 2 * frozen_spread)
    speedup = round(ref_total / c, 4)
    perf = {
        "mode": "lite_vs_frozen",
        "reference_s_per_iteration": frozen["s_per_iteration"],
        "candidate_s_per_iteration": cand["s_per_iteration"],
        "speedup": speedup,
        "frozen_reference_spread": frozen_spread,
        "runs": runs,
        "notes": notes,
        "benefit": {
            "speedup": speedup,
            "speedup_bar": round(bar, 4),
            "peak_memory_allocated_gib": cand["peak_memory_allocated_gib"],
            "reference_peak_memory_allocated_gib": frozen.get("peak_memory_allocated_gib"),
            "passed": speedup > bar,
        },
    }
    return ("ok" if perf["benefit"]["passed"] else "UNAVAILABLE"), perf


def benefit_final(tree, out, cand_env):
    """Final check Gate 1: reference_a -> candidate -> reference_b in 3 independent processes."""
    t = CFG["timing"]
    frozen = json.loads((ROOT / "task" / "timing_ref.json").read_text())
    ref_src = check_reference()
    with scratch_dir("vggt-ref-") as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / "reference")
        ref_a = time_tree(ref_tree, out, "ref_a")
        cand = time_tree(tree, out, "cand", cand_env=cand_env) if ref_a["status"] == "ok" else None
        ref_b = time_tree(ref_tree, out, "ref_b") if cand and cand["status"] == "ok" else None
    runs = {"ref_a": strip(ref_a), "cand": strip(cand), "ref_b": strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v["status"] != "ok"}
    if bad:
        status = "INVALID" if any(v and v["status"] in ("INVALID", "TIMEOUT") for v in bad.values()) else "ERROR"
        return status, {"speedup": None, "runs": runs}
    diffs = {k: fingerprint_diff(ref_a.get("fingerprint"), v.get("fingerprint")) for k, v in runs.items() if k != "ref_a"}
    for k in diffs:
        diffs[k] = [d for d in diffs[k] if d != "candidate_environment"]
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return "ERROR", {
            "speedup": None,
            "runs": runs,
            "fingerprint_diffs": diffs,
            "reason": f"timing runs not comparable: fingerprint differs from ref_a in {badfp}",
        }
    ra, rb, c = ref_a["total_s"], ref_b["total_s"], cand["total_s"]
    ref = (ra + rb) / 2
    perf = {
        "mode": "final_bab",
        "reference_s_per_iteration": round(ref / ref_a["iterations_timed"], 4),
        "candidate_s_per_iteration": cand["s_per_iteration"],
        "speedup": round(ref / c, 4),
        "reference_spread": round(abs(ra - rb) / ref, 4),
        "frozen_reference_s_per_iteration": frozen["s_per_iteration"],
        "runs": runs,
        "fingerprint_diffs": diffs,
    }
    problems = []
    for name, run in (("ref_a", ref_a), ("ref_b", ref_b)):
        off = run["total_s"] / frozen["total_s"] - 1
        if abs(off) > t["frozen_band"]:
            problems.append(
                f'{name} took {run["total_s"]:.1f}s, {off:+.1%} off frozen reference {frozen["total_s"]:.1f}s '
                f'(> {t["frozen_band"]:.0%})'
            )
    pair = [abs(a - b) / ((a + b) / 2) for a, b in zip(ref_a["window_s"], ref_b["window_s"])]
    perf["reference_pair_window_spread"] = [round(x, 4) for x in pair]
    if max(pair) > t["pair_band"]:
        problems.append(f'the two reference runs differ by {max(pair):.1%} in a window (> {t["pair_band"]:.0%})')
    for name, run in (("ref_a", ref_a), ("cand", cand), ("ref_b", ref_b)):
        band = t["window_band"] if name != "cand" else t["window_band_candidate"]
        dev = run.get("window_ratio_vs_frozen")
        if dev is not None and abs(dev) > band:
            problems.append(
                f'{name}: window ratio {run["window_ratio"]} is {dev:+.1%} off frozen ratio '
                f'{frozen["window_ratio"]} (> {band:.0%})'
            )
    perf["problems"] = problems
    if problems:
        return "INVALID", perf
    bar = max(CFG["benefit"]["min_speedup"], 1 + 2 * perf["reference_spread"])
    perf["benefit"] = {
        "speedup": perf["speedup"],
        "speedup_bar": round(bar, 4),
        "peak_memory_allocated_gib": cand["peak_memory_allocated_gib"],
        "reference_peak_memory_allocated_gib": round(
            (ref_a["peak_memory_allocated_gib"] + ref_b["peak_memory_allocated_gib"]) / 2, 2
        ),
        "passed": perf["speedup"] > bar,
    }
    return ("ok" if perf["benefit"]["passed"] else "UNAVAILABLE"), perf


# ---------------------------------------------------------------- verdict

def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False, n1=False, final=False):
    cand_env = cand_env or {}
    if final and not (ROOT / "task" / "final-evaluation").exists():
        return "ERROR", {"passed": False, "stage": "final_marker", "note": "check --final requires task/final-evaluation"}, None
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    if n1:
        traj = trajectories(tree, out, cand_env)
        correctness = {"stage": "statistical (n1 only)", "trajectory": traj}
        if traj["verdict"] == "equivalent":
            return "PASS", dict(correctness, passed=True, verdict="equivalent"), None
        if traj["verdict"] == "error":
            return ("INVALID" if traj.get("status") == "INVALID" else "ERROR"), dict(correctness, passed=False), None
        return "INCORRECT", dict(correctness, passed=False, verdict=traj["verdict"]), None
    if quick:
        num = numerics(tree, out, cand_env)
        if num.get("exact_failures"):
            return "INCORRECT", {"passed": False, "stage": "structure", "verdict": "structure", "numerics": num}, None
        if num.get("errors") or num.get("status") not in ("PASS", "INCORRECT"):
            return "ERROR", {"passed": False, "stage": "structure", "numerics": num}, None
        return (
            "PASS",
            {
                "passed": True,
                "stage": "structure (quick: gates 1 and 3 not run)",
                "bitwise": num["bitwise"],
                "measured_tier": num.get("measured_tier"),
                "numerics": num,
            },
            None,
        )
    performance = None
    if not skip_timing:
        status, performance = (benefit_final if final else benefit_lite)(tree, out, cand_env)
        if status in ("ERROR", "INVALID"):
            return (
                status,
                {
                    "passed": False,
                    "stage": "benefit",
                    "note": (performance or {}).get("reason")
                    or "; ".join((performance or {}).get("problems") or [])
                    or "timing failed",
                },
                performance,
            )
        if status == "UNAVAILABLE":
            performance["measured_speedup"] = performance.pop("speedup")
            return (
                "UNAVAILABLE",
                {
                    "passed": None,
                    "stage": "benefit",
                    "verdict": "no_benefit",
                    "note": "not faster than reference: rejected without precision evaluation",
                },
                performance,
            )
    num = numerics(tree, out, cand_env)
    tier = num.get("measured_tier")
    correctness = {"stage": "bitwise", "measured_tier": tier, "numerics": num}
    if num.get("exact_failures"):
        return "INCORRECT", dict(correctness, passed=False, verdict="structure", note="exact checks failed"), performance
    if num.get("errors") or num.get("status") not in ("PASS", "INCORRECT"):
        return "ERROR", dict(correctness, passed=False, note="numerics could not compare records"), performance
    if tier not in TIERS or TIERS.index(tier) > TIERS.index(CFG["agent_max_tier"]):
        return (
            "INCORRECT",
            dict(correctness, passed=False, verdict="needs_signoff", note=f'measured tier {tier} exceeds {CFG["agent_max_tier"]}'),
            performance,
        )
    if num["bitwise"]:
        return "PASS", dict(correctness, passed=True, verdict="bitwise"), performance
    traj = trajectories(tree, out, cand_env)
    correctness.update(stage="statistical", trajectory=traj)
    if traj["verdict"] == "equivalent":
        return "PASS", dict(correctness, passed=True, verdict="equivalent"), performance
    if traj["verdict"] == "error":
        return (
            ("INVALID" if traj.get("status") == "INVALID" else "ERROR"),
            dict(correctness, passed=False, verdict="error"),
            performance,
        )
    return (
        "INCORRECT",
        dict(
            correctness,
            passed=False,
            verdict=traj["verdict"],
            note=f'endpoint bias exceeds {CFG["trajectory"].get("sigma_k", 3)} x frozen reference seed sigma',
        ),
        performance,
    )


def guarded(out, fn, tree_src=None):
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs("start")
    if not start["ok"]:
        return {
            "status": "ERROR",
            "stage": "inputs",
            "inputs": [start],
            "note": "a frozen input differs from task/formulation.json; nothing was measured",
        }
    if tree_src is None:
        res = fn(None)
    else:
        with scratch_dir("vggt-src-") as tmp:
            tree = materialize(tree_src, Path(tmp) / "tree")
            digest = tree_digest(tree)
            try:
                res = fn(tree)
            except SystemExit as exc:
                res = {"status": "ERROR", "note": f"evaluation stopped: {exc}"}
            res["source_digest"] = digest
            if tree_digest(tree) != digest:
                res["status"] = "ERROR"
                res["note"] = "the tree changed while being measured"
    end = verify_inputs("end")
    res["inputs"] = [start, end]
    if not end["ok"]:
        res["status"] = "ERROR"
        res["note"] = "a frozen input changed during the run"
    return res


def finish(out, name, res):
    res["cost"] = cost()
    res["evaluator_sha256"] = evaluator_sha256()
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / name).write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f'wrote {out / name}: status {res.get("status")}')
    summary = {"status": res.get("status")}
    perf = res.get("performance")
    if isinstance(perf, dict):
        summary["speedup"] = perf.get("speedup", perf.get("measured_speedup"))
    for k in ("speedup_vs_frozen_reference", "s_per_iteration"):
        if k in res:
            summary[k] = res[k]
    print(json.dumps(summary))
    return 0 if res.get("status") in ("PASS", "ok") else 1


def cmd_check(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        status, correctness, performance = evaluate(
            tree,
            out,
            skip_timing=a.skip_timing,
            cand_env=cand_env,
            quick=a.quick,
            n1=a.n1,
            final=a.final,
        )
        return {
            "status": status,
            "final": bool(a.final),
            "n1": bool(a.n1),
            "tree": str(Path(a.tree).resolve()),
            "config": a.config,
            "environment": cand_env,
            "correctness": correctness,
            "performance": performance,
        }

    return finish(out, "check.json", guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        res = strip(time_tree(tree, out, "tree", cand_env=cand_env))
        ref_file = ROOT / "task" / "timing_ref.json"
        if ref_file.exists():
            frozen = json.loads(ref_file.read_text())
            if res["status"] == "ok":
                res["speedup_vs_frozen_reference"] = round(frozen["total_s"] / res["total_s"], 4)
            res["frozen_reference_s_per_iteration"] = frozen["s_per_iteration"]
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)

    return finish(out, "time.json", guarded(out, run, Path(a.tree).resolve()))


CATEGORIES = [
    ("nccl", r"nccl"),
    ("attention", r"fmha|flash|attention|mem_eff|cutlass.*attention|cudnn.*sdpa"),
    ("gemm", r"gemm|cutlass|sm90_xmma|sm80_xmma|cublas|ampere_|hopper_|s16816|nvjet|splitk"),
    ("conv", r"conv|implicit_gemm|winograd|cudnn|dgrad|wgrad|fprop|scudnn"),
    ("norm", r"norm"),
    ("optimizer", r"adam|multi_tensor|foreach"),
    ("memcpy/cast", r"copy|memcpy|cast|convert|_to_copy|memset"),
    ("triton", r"^triton_"),
    ("elementwise/reduce", r"elementwise|reduce|vectorized|unrolled|index|cat|silu|gelu|softmax|mul|add|kthvalue|sort|upsample|bilinear"),
]


def categorize(kernels):
    cats = {}
    for k in kernels:
        n = k["name"].lower()
        cat = next((c for c, rx in CATEGORIES if re.search(rx, n)), "other")
        cats[cat] = cats.get(cat, 0.0) + k["total_ms"]
    return {k: round(v, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    mode = "nsys" if a.nsys else "torch"

    def run(tree):
        res = time_tree(tree, out, "profile", cand_env=cand_env, profile=mode, profile_iters=a.iters)
        ranks = res.pop("_ranks", None)
        if res["status"] != "ok":
            return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
        res["note"] = f'{a.iters} iterations after warmup of {CFG["timing"]["warmup"]}; instrumented (not comparable with `time`)'
        r0 = ranks[0]
        ends = r0["iteration_end_s"]
        first = CFG["timing"]["warmup"]
        res["iteration_s_host"] = [round(ends[i] - ends[i - 1], 3) for i in range(first, min(len(ends), first + a.iters))]
        if mode == "torch":
            summ = json.loads((out / "timing_profile.torchprof.json").read_text())
            summ["categories_ms"] = categorize(summ["kernels"])
            res["torch_profile"] = summ
            trace_path = out / "timing_profile.trace.json"
            res["trace"] = str(trace_path)
            # Run kernel-utilization-analysis next to the trace (per FORMULATOR_BRIEF.md section 3)
            ku = BENCH / "kernel_util.py"
            if ku.exists() and trace_path.exists():
                kmd = out / "kernels.md"
                kjs = out / "kernels.json"
                subprocess.run(
                    [str(PY), str(ku), str(trace_path), "--out", str(kmd), "--json", str(kjs)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                if kmd.exists():
                    res["kernel_util_md"] = str(kmd)
                    res["kernel_util_json"] = str(kjs)
            md = [
                "# profile",
                "",
                f"tree {a.tree}, config {a.config}",
                "",
                res["note"],
                "",
                f'window {summ["span_s"]:.2f} s for {summ["iterations"]} iterations; GPU busy '
                f'{summ["gpu_busy_s"]:.2f} s ({summ["gpu_busy_fraction"]:.0%}); kernel time {summ["kernel_time_s"]:.2f} s (rank 0)',
                "",
                "host time per iteration (s): " + ", ".join(map(str, res["iteration_s_host"])),
                "",
                "## kernel time by category (ms, rank 0)",
                "",
                "| category | ms |",
                "|---|---|",
            ]
            md += [f"| {k} | {v} |" for k, v in summ["categories_ms"].items()]
            md += [
                "",
                "## GPU time inside model calls (forward only)",
                "",
                "| call | GPU ms | calls | CPU ms |",
                "|---|---|---|---|",
            ]
            md += [f'| {k} | {v["cuda_ms"]} | {v["calls"]} | {v["cpu_ms"]} |' for k, v in summ["annotations"].items()]
            md += ["", "## top kernels", "", "| ms | calls | share | kernel |", "|---|---|---|---|"]
            md += [
                f'| {k["total_ms"]} | {k["calls"]} | {k["share_of_kernel_time"]} | `{k["name"][:140]}` |'
                for k in summ["kernels"][:40]
            ]
            md += ["", "## top CPU ops (self time)", "", "| ms | calls | op |", "|---|---|---|"]
            md += [f'| {o["self_cpu_ms"]} | {o["calls"]} | {o["name"]} |' for o in summ["cpu_ops_self_time"][:25]]
            (out / "profile.md").write_text("\n".join(md) + "\n")
        else:
            res["trace"] = str(out / "nsys_profile.nsys-rep")
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)

    return finish(out, "profile.json", guarded(out, run, Path(a.tree).resolve()))


def cmd_baseline(a):
    out = Path(a.out).resolve()

    def run(_):
        ref_src = check_reference()
        with scratch_dir("vggt-ref-") as tmp:
            tree = materialize(ref_src, Path(tmp) / "reference")
            runs = {k: time_tree(tree, out, k) for k in ("ref_a", "ref_b", "ref_c")}
        ok = [v for v in runs.values() if v["status"] == "ok"]
        tot = [v["total_s"] for v in ok]
        ratios = [v["window_ratio"] for v in ok if v.get("window_ratio") is not None]
        res = {
            "status": "ok" if len(ok) == len(runs) else "ERROR",
            "runs": {k: strip(v) for k, v in runs.items()},
            "total_s_runs": tot,
            "total_s": round(statistics.mean(tot), 3) if tot else None,
            "s_per_iteration": round(statistics.mean(v["s_per_iteration"] for v in ok), 4) if ok else None,
            "window_ratio": round(statistics.mean(ratios), 4) if ratios else None,
            "peak_memory_allocated_gib": round(statistics.mean(v["peak_memory_allocated_gib"] for v in ok), 2) if ok else None,
            "spread": round((max(tot) - min(tot)) / statistics.mean(tot), 4) if tot else None,
        }
        if ok:
            r = ok[0]["_ranks"]
            res["rng_fingerprints"] = {str(x["rank"]): x["rng_fingerprints"] for x in r}
            res["rng_identical_across_runs"] = all(
                [x["rng_fingerprints"] for x in v["_ranks"]] == [x["rng_fingerprints"] for x in r] for v in ok
            )
        return res

    return finish(out, "baseline.json", guarded(out, run))


def cmd_record(a):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text()) if a.cases else json.loads(path(CFG["numerics"]["cases"]).read_text())
    tree = Path(a.tree).resolve()
    if a.traj_seeds:
        spec = json.loads(path(CFG["trajectory"]["spec"]).read_text())
        cases = [traj_case(spec, int(s)) for s in a.traj_seeds.split(",")]
    with scratch_dir("vggt-recsrc-") as tmp:
        mtree = materialize(tree, Path(tmp) / "tree")
        failed = record_cases(
            mtree,
            out,
            cases,
            out,
            candidate_environment(a.config),
            "formulate",
            variant=a.variant,
            losses_only=bool(a.traj_seeds),
        )
    print(json.dumps({"status": "ok" if not failed else failed, "out": str(out), "cost": cost()}))
    return 1 if failed else 0


def cmd_freeze_inputs(a):
    inputs = freeze_inputs(a.paths)
    f = ROOT / "task" / "formulation.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(f.read_text()) if f.exists() else {}
    data["inputs"] = inputs
    data["evaluator_sha256"] = evaluator_sha256()
    f.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({"files": len(inputs), "sha256": sum(v["kind"] == "sha256" for v in inputs.values()), "evaluator_sha256": data["evaluator_sha256"]}))
    return 0


def cmd_loader(a):
    got, err = loader_digest(Path(a.tree).resolve())
    print(json.dumps(got) if got else err)
    return 0 if got else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("check")
    k.add_argument("--tree", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--config")
    k.add_argument("--quick", action="store_true")
    k.add_argument("--skip-timing", action="store_true")
    k.add_argument("--n1", action="store_true")
    k.add_argument("--final", action="store_true", help=argparse.SUPPRESS)
    t = sub.add_parser("time")
    t.add_argument("--tree", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config")
    f = sub.add_parser("profile")
    f.add_argument("--tree", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--config")
    f.add_argument("--nsys", action="store_true")
    f.add_argument("--iters", type=int, default=5)
    b = sub.add_parser("baseline")
    b.add_argument("--out", required=True)
    r = sub.add_parser("record", help="formulator: record numerics cases or reference trajectories")
    r.add_argument("--tree", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cases")
    r.add_argument("--variant")
    r.add_argument("--config")
    r.add_argument("--traj-seeds")
    ld = sub.add_parser("loader", help="formulator: print dataset digest of a tree")
    ld.add_argument("--tree", required=True)
    fi = sub.add_parser("freeze-inputs", help="formulator: write frozen input fingerprints and evaluator hash")
    fi.add_argument("paths", nargs="+")
    a = p.parse_args()
    return {
        "freeze-inputs": cmd_freeze_inputs,
        "check": cmd_check,
        "time": cmd_time,
        "profile": cmd_profile,
        "baseline": cmd_baseline,
        "record": cmd_record,
        "loader": cmd_loader,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
