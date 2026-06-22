import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from src.utils.logging import get_logger
from .enhancement import CLAHEEnhancer, BandpassFilter
from .normalization import PercentileNormalizer

logger = get_logger("preprocessing.pipeline")

@dataclass
class PreprocessingResult:
    image: np.ndarray
    original: np.ndarray
    clahe_applied: bool
    bandpass_applied: bool
    normalized: bool
    processing_time: float = 0.0

class PreprocessingPipeline:
    def __init__(self,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid: Tuple[int, int] = (8, 8),
                 bandpass_low_sigma: float = 50.0,
                 bandpass_high_sigma: float = 1.0,
                 percentile_low: float = 0.1,
                 percentile_high: float = 99.9,
                 patch_size: int = 512,
                 overlap: int = 64,
                 apply_clahe: bool = True,
                 apply_bandpass: bool = True):
        self.patch_size = patch_size
        self.overlap = overlap
        self.apply_clahe = apply_clahe
        self.apply_bandpass = apply_bandpass
        self.clahe = CLAHEEnhancer(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid
        ) if apply_clahe else None
        self.bandpass = BandpassFilter(
            low_sigma=bandpass_low_sigma,
            high_sigma=bandpass_high_sigma
        ) if apply_bandpass else None
        self.normalizer = PercentileNormalizer(
            percentile_low=percentile_low,
            percentile_high=percentile_high
        )
        logger.info(f"Preprocessing pipeline initialized: "
                    f"CLAHE={apply_clahe}, Bandpass={apply_bandpass}, "
                    f"patch_size={patch_size}, overlap={overlap}")

    def _extract_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        h, w = image.shape
        step = self.patch_size - self.overlap
        patches = []
        positions = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                y_start = max(0, y_end - self.patch_size)
                x_start = max(0, x_end - self.patch_size)
                patch = image[y_start:y_end, x_start:x_end]
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    padded = np.zeros((self.patch_size, self.patch_size), dtype=patch.dtype)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((x_start, y_start, x_end, y_end))
        return patches, positions

    def _merge_patches(self, patches: List[np.ndarray],
                       positions: List[Tuple[int, int, int, int]],
                       output_shape: Tuple[int, int]) -> np.ndarray:
        h, w = output_shape
        merged = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        half_overlap = self.overlap // 2
        for patch, (x_start, y_start, x_end, y_end) in zip(patches, positions):
            ph, pw = patch.shape[:2]
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            patch_cropped = patch[:actual_h, :actual_w]
            weight = np.ones((actual_h, actual_w), dtype=np.float32)
            if half_overlap > 0:
                fade_y = np.ones(actual_h, dtype=np.float32)
                fade_x = np.ones(actual_w, dtype=np.float32)
                if y_start > 0:
                    fade_y[:half_overlap] = np.linspace(0, 1, half_overlap)
                if y_end < h:
                    fade_y[-half_overlap:] = np.linspace(1, 0, half_overlap)
                if x_start > 0:
                    fade_x[:half_overlap] = np.linspace(0, 1, half_overlap)
                if x_end < w:
                    fade_x[-half_overlap:] = np.linspace(1, 0, half_overlap)
                weight = fade_y[:, np.newaxis] * fade_x[np.newaxis, :]
            merged[y_start:y_end, x_start:x_end] += patch_cropped * weight
            weights[y_start:y_end, x_start:x_end] += weight
        weights[weights == 0] = 1
        return merged / weights

    def _process_single(self, image: np.ndarray) -> np.ndarray:
        import time
        start_time = time.time()
        result = image.astype(np.float32)
        if self.apply_bandpass and self.bandpass is not None:
            result = self.bandpass(result)
        if self.apply_clahe and self.clahe is not None:
            result = self.clahe(result)
        result = self.normalizer(result)
        elapsed = time.time() - start_time
        logger.debug(f"Single image preprocessed in {elapsed*1000:.2f}ms")
        return result

    def process(self, image: np.ndarray, use_patching: bool = False) -> PreprocessingResult:
        import time
        start_time = time.time()
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[-1] == 1:
                image = image.squeeze(-1)
        original = image.copy()
        h, w = image.shape
        if use_patching and (h > self.patch_size or w > self.patch_size):
            patches, positions = self._extract_patches(image)
            processed_patches = [self._process_single(p) for p in patches]
            processed = self._merge_patches(processed_patches, positions, (h, w))
        else:
            processed = self._process_single(image)
        elapsed = time.time() - start_time
        return PreprocessingResult(
            image=processed,
            original=original,
            clahe_applied=self.apply_clahe,
            bandpass_applied=self.apply_bandpass,
            normalized=True,
            processing_time=elapsed
        )

    def process_batch(self, images: np.ndarray, use_patching: bool = False) -> List[PreprocessingResult]:
        results = []
        for i in range(images.shape[0]):
            img = images[i]
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            results.append(self.process(img, use_patching))
        return results

    def __call__(self, image: np.ndarray, use_patching: bool = False) -> PreprocessingResult:
        return self.process(image, use_patching)

    def get_config(self) -> Dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "overlap": self.overlap,
            "apply_clahe": self.apply_clahe,
            "apply_bandpass": self.apply_bandpass,
            "clahe_clip_limit": self.clahe.clip_limit if self.clahe else None,
            "clahe_tile_grid": self.clahe.tile_grid_size if self.clahe else None,
            "bandpass_low_sigma": self.bandpass.low_sigma if self.bandpass else None,
            "bandpass_high_sigma": self.bandpass.high_sigma if self.bandpass else None,
        }
