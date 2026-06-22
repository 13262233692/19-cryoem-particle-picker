import numpy as np
from typing import Tuple, Optional
from src.utils.logging import get_logger

logger = get_logger("preprocessing.normalization")

class PercentileNormalizer:
    def __init__(self, percentile_low: float = 0.1, percentile_high: float = 99.9,
                 target_mean: float = 0.0, target_std: float = 1.0):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.target_mean = target_mean
        self.target_std = target_std
        logger.info(f"Percentile normalizer initialized: {percentile_low}-{percentile_high}%")

    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        data = image[mask] if mask is not None else image
        p_low = np.percentile(data, self.percentile_low)
        p_high = np.percentile(data, self.percentile_high)
        clipped = np.clip(image, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
        normalized = normalized * self.target_std + self.target_mean
        return normalized.astype(np.float32)

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.apply(image, mask)

class ZScoreNormalizer:
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        self.mean = mean
        self.std = std
        logger.info("Z-score normalizer initialized")

    def apply(self, image: np.ndarray) -> np.ndarray:
        mean = self.mean if self.mean is not None else np.mean(image)
        std = self.std if self.std is not None else np.std(image)
        normalized = (image - mean) / (std + 1e-8)
        return normalized.astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

class MinMaxNormalizer:
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None,
                 feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.min_val = min_val
        self.max_val = max_val
        self.feature_range = feature_range
        logger.info(f"Min-max normalizer initialized: range={feature_range}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        min_val = self.min_val if self.min_val is not None else np.min(image)
        max_val = self.max_val if self.max_val is not None else np.max(image)
        scale = self.feature_range[1] - self.feature_range[0]
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        normalized = normalized * scale + self.feature_range[0]
        return normalized.astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

class RobustNormalizer:
    def __init__(self, median: Optional[float] = None, mad: Optional[float] = None):
        self.median = median
        self.mad = mad
        logger.info("Robust normalizer initialized")

    def apply(self, image: np.ndarray) -> np.ndarray:
        median = self.median if self.median is not None else np.median(image)
        if self.mad is not None:
            mad = self.mad
        else:
            mad = np.median(np.abs(image - median))
        normalized = (image - median) / (1.4826 * mad + 1e-8)
        return normalized.astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)
