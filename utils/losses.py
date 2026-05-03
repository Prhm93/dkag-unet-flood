"""BCE + Dice combined loss for binary segmentation."""
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Weighted sum of BCE-with-logits and (soft) Dice loss.

    Parameters
    ----------
    bce_weight : float, default 0.5
        Weight on the BCE term. The Dice term gets `1 - bce_weight`.
    smooth : float, default 1e-5
        Numerical-stability epsilon for the Dice denominator/numerator.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth     = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        prob   = torch.sigmoid(logits)
        inter  = (prob * targets).sum(dim=(2, 3))
        union  = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2 * inter + self.smooth) / (union + self.smooth)

        return (
            self.bce_weight * bce_loss
            + (1 - self.bce_weight) * dice_loss.mean()
        )
