"""
Central configuration for DKAG U-Net training.

All hyper-parameters live here so experiments are easy to reproduce. Override
any of these from the command line via `train.py --flag value`.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    img_dir: str  = "data/images"
    mask_dir: str = "data/masks"
    save_dir: str = "runs/exp1"

    # ── Image / data ─────────────────────────────────────────────────────────
    img_size: int    = 256
    val_split: float = 0.2

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int    = 8
    num_epochs: int    = 60
    lr: float          = 1e-4
    weight_decay: float = 1e-5
    bce_weight: float  = 0.5      # weight on BCE in BCE+Dice loss
    grad_clip: float   = 5.0
    num_workers: int   = 2

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Misc ─────────────────────────────────────────────────────────────────
    # ImageNet normalisation (ResNet34 was trained with these stats)
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Models to train. One of: "standard", "attention", "dkag", or "all".
    model: str = "all"


# A single shared default. Mutate fields directly or build a fresh Config()
# in your scripts.
default_cfg = Config()
