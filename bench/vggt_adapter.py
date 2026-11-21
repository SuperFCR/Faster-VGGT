"""numerics adapter for VGGT training (facebookresearch/vggt @ a288dd0f, training/launch.py + training/trainer.py).

build() reproduces training/launch.py: writes the frozen Hydra config into the materialized tree,
composes it with Hydra, constructs Trainer(**cfg) from the tree under test, and runs the pre-loop
setup of Trainer.run_train() + Trainer.train_epoch() (set_seeds, get_loader, model.train(),
gradient_clipper.setup_clipping).

step() extracts the verbatim AST body of `for data_iter, batch in enumerate(train_loader):` from the
tree's training/trainer.py and executes it for one training iteration. Gate 0 of bench.py verifies
that launch.py (`main`, `__main__`) and trainer.py (`run`, `run_train`, `train_epoch` outside that
per-step loop body) match the reference AST.
"""
import ast
import gc
import hashlib
import importlib
import os
import sys
import time
from pathlib import Path

import torch

from numerics.adapter import Adapter, Objects
from bench import recipe

WRAPPER_PREFIXES = ("module.", "_orig_mod.", "_checkpoint_wrapped_module.")


def strip(name):
    for p in WRAPPER_PREFIXES:
        name = name.replace(p, "")
    return name


def _digest_batch(batch):
    h = hashlib.sha256()
    for k, v in sorted(batch.items(), key=lambda kv: str(kv[0])):
        h.update(str(k).encode())
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode() + str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()[:16]


class Feed:
    """Wraps the trainer's deterministic DataLoader iterator and hashes every batch handed out."""

    def __init__(self, it):
        self.it = it
        self.digests = []
        self.shapes = []

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.it)
        self.digests.append(_digest_batch(batch))
        imgs = batch.get("images")
        self.shapes.append(list(imgs.shape) if isinstance(imgs, torch.Tensor) else None)
        return batch


def _loop_body(trainer_py):
    """Extract the verbatim body of `for data_iter, batch in enumerate(train_loader):` in Trainer.train_epoch()."""
    tree = ast.parse(Path(trainer_py).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Trainer":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "train_epoch":
                    for inner in ast.walk(fn):
                        if (
                            isinstance(inner, ast.For)
                            and isinstance(inner.target, ast.Tuple)
                            and len(inner.target.elts) == 2
                            and isinstance(inner.target.elts[0], ast.Name)
                            and inner.target.elts[0].id == "data_iter"
                        ):
                            wrapped = ast.For(
                                target=ast.Name(id="_rba_once", ctx=ast.Store()),
                                iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
                                body=inner.body,
                                orelse=[],
                            )
                            module = ast.Module(body=[wrapped], type_ignores=[])
                            ast.fix_missing_locations(module)
                            return compile(module, filename=str(trainer_py), mode="exec")
    raise RuntimeError(f"could not find the per-step loop of Trainer.train_epoch() in {trainer_py}")


def _scalar(v):
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return v.detach().reshape(()).to(torch.float64)
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return torch.tensor(float(v), dtype=torch.float64)
    return None


class VggtAdapter(Adapter):
    name = "vggt_vkitti"

    def __init__(self, repo, vkitti, ckpt, scratch, variant=None):
        self.repo = Path(repo).resolve()
        self.vkitti = str(vkitti)
        self.ckpt = str(ckpt)
        self.scratch = Path(scratch)
        if variant not in (None, "tf32_off", "sdpa_mem_efficient", "sdpa_math", "nondeterministic", "lr_control"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.trainer = None
        self.case = {}
        self._handles = []
        self._step_losses = []
        self._feed = None
        self._last_digest = None
        self._last_shape = None
        self._ns = None
        self._code = None

    def configure(self, case):
        self.case = dict(case)

    def build(self, seed, golden=False):
        if golden:
            raise SystemExit("VGGT-1B has ~1.12B parameters; no fp64 golden record")
        det = torch.are_deterministic_algorithms_enabled()
        if det and self.variant != "nondeterministic":
            # In deterministic mode, enable strict deterministic algorithms where supported.
            # NOTE: Checked at formulation time whether warn_only=False or True is required for VGGT ops
            # (e.g., upsample_bilinear2d_backward / boolean mask scatter / flash SDPA).
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Route SDPA to math in deterministic mode for both reference and candidate alike if flash/mem_eff
            # is nondeterministic; controlled by bench case flag `det_sdpa_math` (default True).
            if self.case.get("det_sdpa_math", True):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            strict = bool(self.case.get("strict_deterministic", True))
            torch.use_deterministic_algorithms(True, warn_only=not strict)

        self._release()
        training_dir = self.repo / "training"
        os.chdir(training_dir)
        for p in (str(self.repo), str(training_dir)):
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, str(self.repo))
        sys.path.insert(0, str(training_dir))

        case_scratch = self.scratch / f'{self.case.get("name", "case")}_s{seed}'
        lr_scale = float(self.case.get("lr_scale", 1.0))
        if self.variant == "lr_control":
            lr_scale *= 1.02

        recipe.write_config(
            self.repo,
            case_scratch,
            self.vkitti,
            self.ckpt,
            config_name=recipe.CONFIG_NAME,
            seed=seed,
            max_epochs=int(self.case.get("max_epochs", 20)),
            limit_train_batches=int(self.case.get("limit_train_batches", 800)),
            limit_val_batches=int(self.case.get("limit_val_batches", 400)),
            lr_scale=lr_scale,
            cudnn_deterministic=(det and self.variant != "nondeterministic"),
            num_workers=int(self.case.get("num_workers", 8)),
            save_freq=0,
        )

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(training_dir / "config")):
            cfg = compose(config_name=recipe.CONFIG_NAME)

        T = importlib.import_module("trainer")
        if Path(T.__file__).resolve() != (training_dir / "trainer.py").resolve():
            raise RuntimeError(f"trainer imported from {T.__file__}, expected {training_dir / 'trainer.py'}")

        self.trainer = T.Trainer(**cfg)
        t = self.trainer

        # Re-apply deterministic / variant math settings after Trainer.__init__ sets backend flags
        if det and self.variant != "nondeterministic":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if self.case.get("det_sdpa_math", True):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            strict = bool(self.case.get("strict_deterministic", True))
            torch.use_deterministic_algorithms(True, warn_only=not strict)

        if self.variant == "tf32_off":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif self.variant == "sdpa_mem_efficient":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        elif self.variant == "sdpa_math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        elif self.variant == "nondeterministic":
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False

        # Replicate pre-loop setup from Trainer.run_train() and Trainer.train_epoch() verbatim
        T.set_seeds(t.seed_value + t.epoch * 100, t.max_epochs, t.distributed_rank)
        train_loader = t.train_dataset.get_loader(epoch=int(t.epoch + t.distributed_rank))
        self._train_loader = train_loader
        self._feed = Feed(iter(train_loader))

        phase = "train"
        batch_time = T.AverageMeter("Batch Time", t.device, ":.4f")
        data_time = T.AverageMeter("Data Time", t.device, ":.4f")
        mem = T.AverageMeter("Mem (GB)", t.device, ":.4f")
        data_times = []
        loss_names = t._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: T.AverageMeter(name, t.device, ":.4f") for name in loss_names}
        for config in t.gradient_clipper.configs:
            param_names = ",".join(config["module_names"])
            loss_meters[f"Grad/{param_names}"] = T.AverageMeter(f"Grad/{param_names}", t.device, ":.4f")
        progress = T.ProgressMeter(
            num_batches=len(train_loader),
            meters=[batch_time, data_time, mem, t.time_elapsed_meter, *loss_meters.values()],
            real_meters={},
            prefix="Train Epoch: [{}]".format(t.epoch),
        )
        t.model.train()
        iters_per_epoch = len(train_loader)
        limit_train_batches = iters_per_epoch if t.limit_train_batches is None else t.limit_train_batches
        if t.gradient_clipper is not None:
            t.gradient_clipper.setup_clipping(t.model)

        self._loss_meters = loss_meters
        self._step_losses = []

        def observe_loss(_mod, _args, output):
            if isinstance(output, dict):
                self._step_losses.append({k: v.detach().clone() if torch.is_tensor(v) else v for k, v in output.items()})

        self._handles = [t.loss.register_forward_hook(observe_loss)]
        # Track trainable head parameters for per-step update-size trajectory endpoints (FORMULATOR_BRIEF §2.7)
        raw_model = self._unwrap(t.model)
        self._head_params = {}
        self._prev_head_params = {}
        for head_name in ("camera_head", "depth_head"):
            mod = getattr(raw_model, head_name, None)
            if mod is not None:
                params = [p for p in mod.parameters() if p.requires_grad]
                self._head_params[head_name] = params
                self._prev_head_params[head_name] = [p.detach().clone() for p in params]
        self._code = _loop_body(training_dir / "trainer.py")
        self._ns = dict(vars(T))
        self._ns.update(
            {
                "self": t,
                "train_loader": train_loader,
                "batch_time": batch_time,
                "data_time": data_time,
                "mem": mem,
                "data_times": data_times,
                "phase": phase,
                "loss_meters": loss_meters,
                "progress": progress,
                "end": time.time(),
                "iters_per_epoch": iters_per_epoch,
                "limit_train_batches": limit_train_batches,
            }
        )

    def _release(self):
        if self.trainer is not None:
            for h in self._handles:
                h.remove()
            self._handles = []
            self._feed = None
            self._train_loader = None
            self.trainer = None
            self._ns = None
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _unwrap(obj):
        for attr in ("optimizer", "module"):
            inner = getattr(obj, attr, None)
            if inner is not None:
                return inner
        return obj

    def objects(self):
        t = self.trainer
        return Objects(
            models={"vggt": self._unwrap(t.model)},
            optimizers={"optim": self._unwrap(t.optims[0])},
            sync={"vggt": "replicated"},
            capture_outputs=("vggt.aggregator", "vggt.camera_head", "vggt.depth_head"),
            count_forwards=True,
        )

    def phases(self):
        return ["train"]

    def set_phase(self, phase, step):
        if phase != "train":
            raise RuntimeError(f"unexpected phase {phase!r} for VGGT")

    def batch(self, step, rank):
        return None  # Trainer's deterministic DataLoader iterator is wrapped in Feed

    def step(self, step, batch):
        t = self.trainer
        self._step_losses = []
        raw_batch = next(self._feed)
        self._last_digest = self._feed.digests[-1]
        self._last_shape = self._feed.shapes[-1]
        steps_before = int(t.steps["train"])

        self._ns["data_iter"] = step
        self._ns["batch"] = raw_batch
        exec(self._code, self._ns)

        if int(t.steps["train"]) != steps_before + t.accum_steps:
            raise RuntimeError(
                f"steps['train'] moved from {steps_before} to {t.steps['train']}, expected +{t.accum_steps}"
            )
        if len(self._step_losses) != t.accum_steps:
            raise RuntimeError(f"loss called {len(self._step_losses)} times, expected {t.accum_steps}")

        # Average micro-batch losses across accum_steps chunks
        losses = {}
        keys = sorted({k for d in self._step_losses for k in d})
        for k in keys:
            vals = [_scalar(d[k]) for d in self._step_losses if k in d]
            vals = [v for v in vals if v is not None]
            if vals:
                losses[k] = torch.stack(vals).mean()

        # Also expose per-group gradient norms recorded by GradientClipper into loss_meters
        for mk, meter in self._loss_meters.items():
            if mk.startswith("Grad/"):
                s = _scalar(meter.val)
                if s is not None:
                    losses[f"grad_norm/{mk[5:]}"] = s

        # Per-step AdamW update magnitude on top-updated elements (linear in lr, tight seed spread; FORMULATOR_BRIEF §2.7)
        with torch.no_grad():
            for head_name, params in getattr(self, "_head_params", {}).items():
                prevs = self._prev_head_params.get(head_name, [])
                if not params:
                    continue
                # Compute per-step |p_t - p_{t-1}| on 1D params (biases/norms) and all params
                bias_deltas = []
                all_deltas_top = []
                for i, (p, p_prev) in enumerate(zip(params, prevs)):
                    d = (p.detach() - p_prev).abs()
                    prevs[i].copy_(p.detach())
                    if p.ndim == 1 and d.numel() > 0:
                        bias_deltas.append(d.reshape(-1))
                    if d.numel() > 0:
                        # Top 1% largest updates in each tensor (Adam full-step regime)
                        k = max(1, min(d.numel(), int(0.01 * d.numel())))
                        all_deltas_top.append(torch.topk(d.reshape(-1), k, sorted=False).values.mean().double())
                if all_deltas_top:
                    losses[f"step_delta_top1/{head_name}"] = torch.stack(all_deltas_top).mean()
                if bias_deltas:
                    cat_b = torch.cat(bias_deltas)
                    k = max(1, int(0.95 * (cat_b.numel() - 1)) + 1)
                    losses[f"step_delta_bias_q95/{head_name}"] = torch.kthvalue(cat_b, k).values.double()

        return losses

    def invariants(self):
        t = self.trainer
        return {
            "steps_train": int(t.steps["train"]),
            "where": round(float(t.where), 10),
            "accum_steps": int(t.accum_steps),
            "batch_digest": self._last_digest,
            "batch_shape": str(self._last_shape),
            "scaler_scale": float(t.scaler.get_scale()) if t.optim_conf.amp.enabled else 1.0,
        }

    def canonical(self, kind, name, tensor):
        return {strip(name): tensor}

    def save_state(self, path):
        raise NotImplementedError("cases start from the frozen published VGGT-1B checkpoint")

    def load_state(self, path):
        raise NotImplementedError("cases start from the frozen published VGGT-1B checkpoint")
