import numpy as np
import cv2
from typing import List, Tuple, Optional
import os

def normalize_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image.squeeze()
    p_low = np.percentile(image, 0.1)
    p_high = np.percentile(image, 99.9)
    image = np.clip(image, p_low, p_high)
    image = (image - p_low) / (p_high - p_low + 1e-8)
    return (image * 255).astype(np.uint8)

def visualize_particles(image: np.ndarray,
                        coordinates: List[Tuple[int, int]],
                        particle_size: int = 32,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    display_img = normalize_for_display(image)
    if display_img.ndim == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    half_size = particle_size // 2
    for (x, y) in coordinates:
        cv2.rectangle(display_img,
                      (int(x - half_size), int(y - half_size)),
                      (int(x + half_size), int(y + half_size)),
                      color, thickness)
    return display_img

def overlay_heatmap(image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    display_img = normalize_for_display(image)
    if display_img.ndim == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (display_img.shape[1], display_img.shape[0]))
    return cv2.addWeighted(display_img, 1 - alpha, heatmap_color, alpha, 0)

def save_result_image(image: np.ndarray,
                      save_path: str,
                      coordinates: Optional[List[Tuple[int, int]]] = None,
                      heatmap: Optional[np.ndarray] = None) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if heatmap is not None:
        result = overlay_heatmap(image, heatmap)
    else:
        result = normalize_for_display(image)
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    if coordinates:
        half_size = 16
        for (x, y) in coordinates:
            cv2.drawMarker(result, (int(x), int(y)), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.rectangle(result,
                          (int(x - half_size), int(y - half_size)),
                          (int(x + half_size), int(y + half_size)),
                          (0, 255, 0), 2)
    cv2.imwrite(save_path, result)
    return save_path
