"""Standard binary segmentation metrics: IoU, Dice, Precision, Recall, Accuracy."""
from typing import Dict

import torch


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute per-batch averaged segmentation metrics.

    Parameters
    ----------
    logits : (B, 1, H, W) float tensor — raw network output (pre-sigmoid).
    targets : (B, 1, H, W) float tensor — binary ground-truth mask.
    threshold : float, default 0.5 — probability threshold for binarisation.
    smooth : float, default 1e-6 — numerical-stability epsilon.

    Returns
    -------
    dict with keys: iou, dice, precision, recall, accuracy.
    Each is the mean over the batch.
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    targ = targets.float()

    TP = (pred * targ).sum(dim=(2, 3))
    FP = (pred * (1 - targ)).sum(dim=(2, 3))
    FN = ((1 - pred) * targ).sum(dim=(2, 3))
    TN = ((1 - pred) * (1 - targ)).sum(dim=(2, 3))

    iou       = (TP + smooth) / (TP + FP + FN + smooth)
    dice      = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall    = (TP + smooth) / (TP + FN + smooth)
    accuracy  = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)

    return {
        "iou":       iou.mean().item(),
        "dice":      dice.mean().item(),
        "precision": precision.mean().item(),
        "recall":    recall.mean().item(),
        "accuracy":  accuracy.mean().item(),
    }
