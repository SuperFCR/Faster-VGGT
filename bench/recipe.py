"""The frozen VGGT training workload (upstream facebookresearch/vggt @ a288dd0f).

Upstream config: training/config/default.yaml (+ default_dataset.yaml), launched with
`torchrun --nproc_per_node=1 launch.py --config _rba_vggt` from `training/`.

Per-GPU load is identical to upstream default.yaml:
  - model: VGGT (enable_camera=True, enable_depth=True, enable_point=False, enable_track=False)
  - pretrained checkpoint: official public facebook/VGGT-1B (model.pt), strict=False
  - frozen_module_names: ["*aggregator*"] (upstream default fine-tuning recipe)
  - img_size: 518, patch_size: 14, max_img_per_gpu: 48, accum_steps: 2, num_workers: 8
  - dataset: public Virtual KITTI 2.0.3 via upstream data.datasets.vkitti.VKittiDataset
  - loss: MultitaskLoss (camera weight=5.0 l1, depth weight=1.0 gradient_loss_fn="grad" valid_range=0.98)
  - optim: torch.optim.AdamW (lr=5e-5, weight_decay=0.05), CompositeParamScheduler (5% linear, 95% cosine),
    gradient_clip max_norm=1.0 per module group (aggregator, depth, camera)
  - precision: fp32 master weights & optimizer state, bf16 autocast (aggregator), fp32 heads (autocast(enabled=False)
    in vggt.py), GradScaler(enabled=True), allow_tf32=True, cudnn_benchmark=False.
"""
from pathlib import Path
import yaml

ENTRY_LAUNCH = "training/launch.py"
ENTRY_TRAINER = "training/trainer.py"
CONFIG_NAME = "_rba_vggt"
WORLD_SIZE = 1


def build_config(
    scratch,
    vkitti_dir,
    ckpt_path,
    *,
    seed=42,
    max_epochs=20,
    limit_train_batches=800,
    limit_val_batches=400,
    lr_scale=1.0,
    cudnn_deterministic=False,
    num_workers=8,
    save_freq=0,
    val_epoch_freq=5,
):
    """Build the frozen Hydra config dict matching upstream training/config/default.yaml."""
    scratch = Path(scratch)
    lr_peak = 5e-5 * float(lr_scale)
    lr_min = 1e-8 * float(lr_scale)

    return {
        "defaults": ["default_dataset.yaml"],
        "exp_name": "rba_vggt",
        "img_size": 518,
        "num_workers": int(num_workers),
        "seed_value": int(seed),
        "accum_steps": 2,
        "patch_size": 14,
        "val_epoch_freq": int(val_epoch_freq),
        "max_img_per_gpu": 48,
        "limit_train_batches": int(limit_train_batches),
        "limit_val_batches": int(limit_val_batches),
        "data": {
            "train": {
                "_target_": "data.dynamic_dataloader.DynamicTorchDataset",
                "num_workers": "${num_workers}",
                "max_img_per_gpu": "${max_img_per_gpu}",
                "common_config": {
                    "img_size": "${img_size}",
                    "patch_size": "${patch_size}",
                    "debug": False,
                    "repeat_batch": False,
                },
                "dataset": {
                    "_target_": "data.composed_dataset.ComposedDataset",
                    "dataset_configs": [
                        {
                            "_target_": "data.datasets.vkitti.VKittiDataset",
                            "split": "train",
                            "VKitti_DIR": str(vkitti_dir),
                            "len_train": 100000,
                            "expand_ratio": 8,
                        }
                    ],
                },
            },
            "val": {
                "_target_": "data.dynamic_dataloader.DynamicTorchDataset",
                "num_workers": "${num_workers}",
                "max_img_per_gpu": "${max_img_per_gpu}",
                "common_config": {
                    "img_size": "${img_size}",
                    "patch_size": "${patch_size}",
                    "debug": False,
                },
                "dataset": {
                    "_target_": "data.composed_dataset.ComposedDataset",
                    "dataset_configs": [
                        {
                            "_target_": "data.datasets.vkitti.VKittiDataset",
                            "split": "test",
                            "VKitti_DIR": str(vkitti_dir),
                            "len_test": 10000,
                            "expand_ratio": 8,
                        }
                    ],
                },
            },
        },
        "logging": {
            "log_dir": str(scratch / "logs"),
            "log_visuals": False,
            "log_freq": 1,
            "log_level_primary": "INFO",
            "log_level_secondary": "WARNING",
            "all_ranks": False,
            "tensorboard_writer": {
                "_target_": "train_utils.tb_writer.TensorBoardLogger",
                "path": "${logging.log_dir}/tensorboard",
            },
            "scalar_keys_to_log": {
                "train": {
                    "keys_to_log": [
                        "loss_objective",
                        "loss_camera",
                        "loss_T",
                        "loss_R",
                        "loss_FL",
                        "loss_conf_depth",
                        "loss_reg_depth",
                        "loss_grad_depth",
                    ]
                },
                "val": {
                    "keys_to_log": [
                        "loss_objective",
                        "loss_camera",
                        "loss_T",
                        "loss_R",
                        "loss_FL",
                        "loss_conf_depth",
                        "loss_reg_depth",
                        "loss_grad_depth",
                    ]
                },
            },
        },
        "checkpoint": {
            "save_dir": str(scratch / "ckpts"),
            "save_freq": int(save_freq),
            "resume_checkpoint_path": str(ckpt_path),
            "strict": False,
        },
        "loss": {
            "_target_": "loss.MultitaskLoss",
            "camera": {
                "weight": 5.0,
                "loss_type": "l1",
            },
            "depth": {
                "weight": 1.0,
                "gradient_loss_fn": "grad",
                "valid_range": 0.98,
            },
            "point": None,
            "track": None,
        },
        "optim": {
            "param_group_modifiers": False,
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": lr_peak,
                "weight_decay": 0.05,
            },
            "frozen_module_names": ["*aggregator*"],
            "amp": {
                "enabled": True,
                "amp_dtype": "bfloat16",
            },
            "gradient_clip": {
                "_target_": "train_utils.gradient_clip.GradientClipper",
                "configs": [
                    {"module_name": ["aggregator"], "max_norm": 1.0, "norm_type": 2},
                    {"module_name": ["depth"], "max_norm": 1.0, "norm_type": 2},
                    {"module_name": ["camera"], "max_norm": 1.0, "norm_type": 2},
                ],
            },
            "options": {
                "lr": [
                    {
                        "scheduler": {
                            "_target_": "fvcore.common.param_scheduler.CompositeParamScheduler",
                            "schedulers": [
                                {
                                    "_target_": "fvcore.common.param_scheduler.LinearParamScheduler",
                                    "start_value": lr_min,
                                    "end_value": lr_peak,
                                },
                                {
                                    "_target_": "fvcore.common.param_scheduler.CosineParamScheduler",
                                    "start_value": lr_peak,
                                    "end_value": lr_min,
                                },
                            ],
                            "lengths": [0.05, 0.95],
                            "interval_scaling": ["rescaled", "rescaled"],
                        }
                    }
                ],
                "weight_decay": [
                    {
                        "scheduler": {
                            "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                            "value": 0.05,
                        }
                    }
                ],
            },
        },
        "max_epochs": int(max_epochs),
        "model": {
            "_target_": "vggt.models.vggt.VGGT",
            "enable_camera": True,
            "enable_depth": True,
            "enable_point": False,
            "enable_track": False,
        },
        "distributed": {
            "backend": "nccl",
            "comms_dtype": None,
            "find_unused_parameters": False,
            "timeout_mins": 30,
            "gradient_as_bucket_view": True,
            "bucket_cap_mb": 25,
            "broadcast_buffers": True,
        },
        "cuda": {
            "cudnn_deterministic": bool(cudnn_deterministic),
            "cudnn_benchmark": False,
            "allow_tf32": True,
        },
    }


def write_config(tree, scratch, vkitti_dir, ckpt_path, config_name=CONFIG_NAME, **kwargs):
    """Write the frozen Hydra config YAML into the materialized tree's training/config/ directory."""
    cfg = build_config(scratch, vkitti_dir, ckpt_path, **kwargs)
    cfg_path = Path(tree) / "training" / "config" / f"{config_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path, cfg
