"""
Main training entry point for the DKAG U-Net repository.

Trains one or more of {Standard U-Net, Attention U-Net (Oktay), DKAG U-Net},
saves the best-IoU checkpoint for each, plots learning curves, and writes a
summary results.json.

Usage
-----
    # Train all three models (default)
    python train.py --img-dir data/images --mask-dir data/masks

    # Train only DKAG
    python train.py --model dkag

    # Override any hyper-parameter
    python train.py --epochs 100 --batch-size 16 --lr 5e-5
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # safe for headless servers
import matplotlib.pyplot as plt
import torch

from config import Config
from models import StandardUNet, AttentionUNet, DKAGUNet
from utils  import get_dataloaders, train_model, set_seed, get_device

warnings.filterwarnings("ignore")


MODEL_REGISTRY = {
    "standard":  ("Standard U-Net",         StandardUNet),
    "attention": ("Attention U-Net",        AttentionUNet),
    "dkag":      ("DKAG U-Net (Proposed)",  DKAGUNet),
}


# ─── CLI parsing ─────────────────────────────────────────────────────────────

def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Train DKAG U-Net and baselines for flood segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--img-dir",     default=cfg.img_dir)
    p.add_argument("--mask-dir",    default=cfg.mask_dir)
    p.add_argument("--save-dir",    default=cfg.save_dir)
    p.add_argument("--img-size",    type=int,   default=cfg.img_size)
    p.add_argument("--batch-size",  type=int,   default=cfg.batch_size)
    p.add_argument("--epochs",      type=int,   default=cfg.num_epochs,
                   dest="num_epochs")
    p.add_argument("--lr",          type=float, default=cfg.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--val-split",   type=float, default=cfg.val_split)
    p.add_argument("--seed",        type=int,   default=cfg.seed)
    p.add_argument("--num-workers", type=int,   default=cfg.num_workers)
    p.add_argument(
        "--model", default=cfg.model,
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Which model to train.",
    )
    args = p.parse_args()

    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Dict]) -> None:
    print(f"\n{'=' * 78}")
    print(
        f"  {'Model':<30} {'IoU':>7} {'Dice':>7} "
        f"{'Prec':>7} {'Recall':>7} {'Acc':>7}"
    )
    print(f"{'=' * 78}")
    for name, m in results.items():
        print(
            f"  {name:<30} "
            f"{m['iou']*100:>6.2f}%  {m['dice']*100:>6.2f}%  "
            f"{m['precision']*100:>6.2f}%  {m['recall']*100:>6.2f}%  "
            f"{m['accuracy']*100:>6.2f}%"
        )
    print(f"{'=' * 78}")


def plot_learning_curves(histories: Dict[str, Dict], save_path: str) -> None:
    """Plot val/train Loss, IoU, Dice for every trained model."""
    if not histories:
        return

    palette = ["#1976D2", "#FF9800", "#4CAF50", "#9C27B0", "#E91E63"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        ("val_loss",  "train_loss", "Loss",       "Loss"),
        ("val_iou",   "train_iou",  "IoU",        "IoU"),
        ("val_dice",  "train_dice", "Dice score", "Dice Score"),
    ]
    for ax, (val_k, train_k, ylabel, title) in zip(axes, panels):
        for i, (name, h) in enumerate(histories.items()):
            color = palette[i % len(palette)]
            ax.plot(h[val_k], label=f"{name} (val)",
                    color=color, linewidth=2)
            ax.plot(h[train_k], label=f"{name} (train)",
                    color=color, linewidth=1.2, ls="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.grid(True, ls="--", alpha=0.4)

    plt.suptitle("DKAG U-Net — Model Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = get_device()
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    set_seed(cfg.seed)
    train_dl, val_dl = get_dataloaders(cfg)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}\n")

    if cfg.model == "all":
        to_train = list(MODEL_REGISTRY.keys())
    else:
        to_train = [cfg.model]

    results:   Dict[str, Dict] = {}
    histories: Dict[str, Dict] = {}

    for key in to_train:
        display_name, ModelCls = MODEL_REGISTRY[key]
        set_seed(cfg.seed)  # reset seed before each training run
        _, hist, metrics = train_model(
            display_name, ModelCls(), train_dl, val_dl, cfg,
        )
        results[display_name]   = metrics
        histories[display_name] = hist

    # ── Summary table ────────────────────────────────────────────────────────
    print_results_table(results)

    # ── Save results.json ────────────────────────────────────────────────────
    results_path = os.path.join(cfg.save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {results_path}")

    # ── Plot learning curves ─────────────────────────────────────────────────
    plot_learning_curves(
        histories, os.path.join(cfg.save_dir, "learning_curves.png")
    )


if __name__ == "__main__":
    main()
