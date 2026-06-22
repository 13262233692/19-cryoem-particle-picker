import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional
from src.utils.logging import get_logger

logger = get_logger("preprocessing.enhancement")

class CLAHEEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        logger.info(f"CLAHE initialized: clip_limit={clip_limit}, tile_grid={tile_grid_size}")

    def _to_uint8(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        p_low = np.percentile(image, 0.1)
        p_high = np.percentile(image, 99.9)
        normalized = np.clip(image, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low + 1e-8)
        return (normalized * 255).astype(np.uint8), p_low, p_high

    def _to_float(self, image: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
        normalized = image.astype(np.float32) / 255.0
        return normalized * (p_high - p_low) + p_low

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[-1] == 1:
                image = image.squeeze(-1)
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")
        img_uint8, p_low, p_high = self._to_uint8(image)
        enhanced = self._clahe.apply(img_uint8)
        return self._to_float(enhanced, p_low, p_high)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

class BandpassFilter:
    def __init__(self, low_sigma: float = 50.0, high_sigma: float = 1.0, use_fft: bool = True):
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.use_fft = use_fft
        logger.info(f"Bandpass filter initialized: low_sigma={low_sigma}, high_sigma={high_sigma}, fft={use_fft}")

    def _gaussian_kernel_fft(self, shape: Tuple[int, int], sigma: float) -> np.ndarray:
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        kernel = np.exp(-dist_sq / (2 * sigma ** 2))
        return kernel.astype(np.float32)

    def _fft_bandpass(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        fft_img = fft2(image.astype(np.float32))
        fft_shifted = fftshift(fft_img)
        low_pass = self._gaussian_kernel_fft((h, w), self.low_sigma)
        high_pass = 1.0 - self._gaussian_kernel_fft((h, w), self.high_sigma)
        bandpass = low_pass * high_pass
        filtered_shifted = fft_shifted * bandpass
        filtered = ifftshift(filtered_shifted)
        result = np.real(ifft2(filtered))
        return result.astype(np.float32)

    def _spatial_bandpass(self, image: np.ndarray) -> np.ndarray:
        low_freq = ndimage.gaussian_filter(image, sigma=self.low_sigma)
        high_freq = ndimage.gaussian_filter(image, sigma=self.high_sigma)
        return high_freq - low_freq

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[-1] == 1:
                image = image.squeeze(-1)
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")
        if self.use_fft:
            result = self._fft_bandpass(image)
        else:
            result = self._spatial_bandpass(image)
        return result.astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

class HighPassFilter:
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        logger.info(f"High-pass filter initialized: sigma={sigma}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        low_freq = ndimage.gaussian_filter(image, sigma=self.sigma)
        return (image - low_freq).astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

class LowPassFilter:
    def __init__(self, sigma: float = 50.0):
        self.sigma = sigma
        logger.info(f"Low-pass filter initialized: sigma={sigma}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(image, sigma=self.sigma).astype(np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)
