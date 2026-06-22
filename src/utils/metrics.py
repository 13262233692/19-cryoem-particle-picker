import numpy as np
from typing import Tuple, List

def compute_dice_coefficient(pred: np.ndarray, target: np.ndarray, 
                             smooth: float = 1e-6) -> float:
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2.0 * intersection + smooth) / (union + smooth)

def compute_precision_recall(pred: np.ndarray, target: np.ndarray,
                             threshold: float = 0.5) -> Tuple[float, float, float]:
    pred = (pred > threshold).astype(np.bool_)
    target = target.astype(np.bool_)
    tp = np.sum(np.logical_and(pred, target))
    fp = np.sum(np.logical_and(pred, np.logical_not(target)))
    fn = np.sum(np.logical_and(np.logical_not(pred), target))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compute_particle_detection_metrics(pred_coords: List[Tuple[int, int]],
                                       gt_coords: List[Tuple[int, int]],
                                       tolerance: int = 5) -> Tuple[float, float, float]:
    matched_gt = set()
    matched_pred = set()
    for i, (px, py) in enumerate(pred_coords):
        for j, (gx, gy) in enumerate(gt_coords):
            if j in matched_gt:
                continue
            dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            if dist <= tolerance:
                matched_pred.add(i)
                matched_gt.add(j)
                break
    tp = len(matched_pred)
    fp = len(pred_coords) - tp
    fn = len(gt_coords) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
