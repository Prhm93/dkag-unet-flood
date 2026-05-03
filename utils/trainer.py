"""
Training and validation loops for any of the three U-Net variants.

`train_model` is the high-level entry point used by `train.py`. It:
  - moves the model to the configured device,
  - sets up AdamW + cosine-annealing,
  - trains for `cfg.num_epochs` with mixed-precision (AMP),
  - saves the best-IoU checkpoint to `cfg.save_dir`,
  - returns the trained model, the per-epoch history, and the
    best-checkpoint validation metrics.
"""
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses  import BCEDiceLoss
from .metrics import compute_metrics


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Per-epoch loops ─────────────────────────────────────────────────────────

def _train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, grad_clip
):
    model.train()
    total_loss = 0.0
    agg = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "accuracy": 0}

    for imgs, masks in tqdm(loader, leave=False, desc="  train"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in compute_metrics(logits.detach(), masks).items():
            agg[k] += v

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in agg.items()}


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    agg = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "accuracy": 0}

    for imgs, masks in tqdm(loader, leave=False, desc="  val  "):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        for k, v in compute_metrics(logits, masks).items():
            agg[k] += v

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in agg.items()}


# ─── High-level entry point ──────────────────────────────────────────────────

def train_model(
    model_name: str,
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg,
) -> Tuple[nn.Module, Dict, Dict]:
    """
    Train one model. Returns (model, history, best_val_metrics).

    The best-IoU checkpoint is saved to
        f"{cfg.save_dir}/{model_name.replace(' ', '_')}_best.pth"
    """
    device = get_device()
    print(f"\n{'=' * 65}")
    print(f"  Training: {model_name}")
    print(f"{'=' * 65}")

    model     = model.to(device)
    criterion = BCEDiceLoss(bce_weight=cfg.bce_weight).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(
        cfg.save_dir, f"{model_name.replace(' ', '_')}_best.pth"
    )

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_iou", "val_iou",
        "train_dice", "val_dice",
        "val_precision", "val_recall", "val_accuracy",
    ]}
    best_iou = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        t_loss, t_m = _train_one_epoch(
            model, train_dl, optimizer, criterion, scaler,
            device, cfg.grad_clip,
        )
        v_loss, v_m = _validate(model, val_dl, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_iou"].append(t_m["iou"])
        history["val_iou"].append(v_m["iou"])
        history["train_dice"].append(t_m["dice"])
        history["val_dice"].append(v_m["dice"])
        history["val_precision"].append(v_m["precision"])
        history["val_recall"].append(v_m["recall"])
        history["val_accuracy"].append(v_m["accuracy"])

        saved_marker = ""
        if v_m["iou"] > best_iou:
            best_iou = v_m["iou"]
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "best_iou":    best_iou,
                    "val_metrics": v_m,
                },
                save_path,
            )
            saved_marker = " ★ SAVED"

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.num_epochs:
            print(
                f"  Ep {epoch:03d}/{cfg.num_epochs} | "
                f"Loss {v_loss:.4f} | IoU {v_m['iou']*100:.2f}% | "
                f"Dice {v_m['dice']*100:.2f}% | "
                f"Prec {v_m['precision']*100:.2f}% | "
                f"Rec {v_m['recall']*100:.2f}% | "
                f"Acc {v_m['accuracy']*100:.2f}% | "
                f"LR {scheduler.get_last_lr()[0]:.2e}{saved_marker}"
            )

    print(f"\n  ✔ Best Val IoU: {best_iou * 100:.2f}%  → {save_path}")

    # Reload best checkpoint for the final reported metrics. Guard against
    # the (very unlikely) case where no epoch ever beat IoU = 0.
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("  [warn] No checkpoint was ever saved — using final epoch state.")
    model.eval()
    _, final_m = _validate(model, val_dl, BCEDiceLoss().to(device), device)

    print(
        f"  [Final best] "
        f"IoU={final_m['iou']*100:.2f}%  "
        f"Dice={final_m['dice']*100:.2f}%  "
        f"Prec={final_m['precision']*100:.2f}%  "
        f"Rec={final_m['recall']*100:.2f}%  "
        f"Acc={final_m['accuracy']*100:.2f}%"
    )
    return model, history, final_m
