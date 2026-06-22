import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.utils.logging import get_logger

logger = get_logger("training.loss")

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] > 1:
            probs = F.softmax(inputs, dim=1)[:, 1, :, :]
        else:
            probs = torch.sigmoid(inputs).squeeze(1)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean", from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            if inputs.shape[1] > 1:
                ce_loss = F.cross_entropy(inputs, targets.long(), reduction="none")
                pt = torch.exp(-ce_loss)
            else:
                inputs = inputs.squeeze(1)
                targets = targets.float()
                ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
                pt = torch.exp(-ce_loss)
        else:
            inputs = inputs.squeeze(1)
            targets = targets.float()
            ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5,
                 smooth: float = 1e-6, focal_gamma: Optional[float] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        if focal_gamma is not None and focal_gamma > 0:
            self.ce_loss = FocalLoss(gamma=focal_gamma)
            logger.info(f"Using FocalLoss with gamma={focal_gamma}")
        else:
            self.ce_loss = nn.CrossEntropyLoss() if 1 > 1 else nn.BCEWithLogitsLoss()
        logger.info(f"DiceCELoss initialized: dice_weight={dice_weight}, ce_weight={ce_weight}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] > 1:
            ce = F.cross_entropy(inputs, targets.long())
        else:
            inputs = inputs.squeeze(1)
            ce = F.binary_cross_entropy_with_logits(inputs, targets.float())
        dice = self.dice_loss(inputs, targets)
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        return total_loss

class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module,
                 deep_weights: Optional[list] = None):
        super().__init__()
        self.base_loss = base_loss
        self.deep_weights = deep_weights

    def forward(self, inputs: list, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, list):
            inputs = [inputs]
        if self.deep_weights is None:
            n = len(inputs)
            weights = [1.0 / (2 ** (n - i - 1)) for i in range(n)]
            total = sum(weights)
            weights = [w / total for w in weights]
        else:
            weights = self.deep_weights
        total_loss = 0.0
        for i, inp in enumerate(inputs):
            if inp.shape[2:] != targets.shape[1:]:
                inp = F.interpolate(inp, size=targets.shape[1:],
                                   mode='bilinear', align_corners=True)
            loss = self.base_loss(inp, targets)
            total_loss += weights[i] * loss
        return total_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] > 1:
            probs = F.softmax(inputs, dim=1)[:, 1, :, :]
        else:
            probs = torch.sigmoid(inputs).squeeze(1)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(1, 2))
        fp = ((1 - targets) * probs).sum(dim=(1, 2))
        fn = (targets * (1 - probs)).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1 - tversky
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
