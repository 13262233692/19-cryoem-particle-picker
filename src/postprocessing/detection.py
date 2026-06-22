import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_filter, label
from typing import List, Tuple, Optional, Dict, Any
from src.utils.logging import get_logger

logger = get_logger("postprocessing.detection")

class PeakDetector:
    def __init__(self,
                 min_distance: int = 20,
                 threshold_abs: float = 0.5,
                 threshold_rel: Optional[float] = None,
                 exclude_border: int = 10,
                 max_particles: int = 10000,
                 footprint_size: int = 3):
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.exclude_border = exclude_border
        self.max_particles = max_particles
        self.footprint_size = footprint_size
        logger.info(f"PeakDetector initialized: min_distance={min_distance}, "
                    f"threshold_abs={threshold_abs}, max_particles={max_particles}")

    def _apply_max_filter(self, prob_map: np.ndarray) -> np.ndarray:
        footprint = np.ones((self.footprint_size, self.footprint_size), dtype=bool)
        return maximum_filter(prob_map, footprint=footprint)

    def detect(self, prob_map: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
        if prob_map.ndim != 2:
            raise ValueError(f"Expected 2D probability map, got shape {prob_map.shape}")
        h, w = prob_map.shape
        if self.threshold_rel is not None:
            threshold = max(self.threshold_abs, self.threshold_rel * prob_map.max())
        else:
            threshold = self.threshold_abs
        prob_map = prob_map.astype(np.float32)
        max_filtered = self._apply_max_filter(prob_map)
        peaks = (prob_map == max_filtered) & (prob_map > threshold)
        if self.exclude_border > 0:
            peaks[:self.exclude_border, :] = False
            peaks[-self.exclude_border:, :] = False
            peaks[:, :self.exclude_border] = False
            peaks[:, -self.exclude_border:] = False
        labeled, num_labels = label(peaks)
        if num_labels == 0:
            return [], []
        centers = []
        scores = []
        for lbl in range(1, num_labels + 1):
            mask = (labeled == lbl)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            score = float(prob_map[cy, cx])
            centers.append((cx, cy))
            scores.append(score)
        if len(centers) > 1 and self.min_distance > 1:
            centers, scores = self._suppress_close_peaks(centers, scores)
        if len(centers) > self.max_particles:
            sort_idx = np.argsort(scores)[::-1][:self.max_particles]
            centers = [centers[i] for i in sort_idx]
            scores = [scores[i] for i in sort_idx]
        return centers, scores

    def _suppress_close_peaks(self,
                              centers: List[Tuple[int, int]],
                              scores: List[float]) -> Tuple[List[Tuple[int, int]], List[float]]:
        if len(centers) == 0:
            return [], []
        sort_idx = np.argsort(scores)[::-1]
        centers_sorted = [centers[i] for i in sort_idx]
        scores_sorted = [scores[i] for i in sort_idx]
        keep = [True] * len(centers_sorted)
        for i in range(len(centers_sorted)):
            if not keep[i]:
                continue
            x1, y1 = centers_sorted[i]
            for j in range(i + 1, len(centers_sorted)):
                if not keep[j]:
                    continue
                x2, y2 = centers_sorted[j]
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < self.min_distance:
                    keep[j] = False
        final_centers = [centers_sorted[i] for i in range(len(centers_sorted)) if keep[i]]
        final_scores = [scores_sorted[i] for i in range(len(centers_sorted)) if keep[i]]
        return final_centers, final_scores

    def detect_with_subpixel(self, prob_map: np.ndarray
                            ) -> Tuple[List[Tuple[float, float]], List[float]]:
        import scipy.optimize as opt
        centers, scores = self.detect(prob_map)
        refined_centers = []
        def gaussian(p, x, y):
            A, x0, y0, sigma_x, sigma_y, bg = p
            return bg + A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))
        def error(p, x, y, z):
            return np.ravel(gaussian(p, x, y) - z)
        for (cx, cy), score in zip(centers, scores):
            window = 3
            y_min, y_max = max(0, cy - window), min(prob_map.shape[0], cy + window + 1)
            x_min, x_max = max(0, cx - window), min(prob_map.shape[1], cx + window + 1)
            local_region = prob_map[y_min:y_max, x_min:x_max]
            if local_region.size >= 9:
                try:
                    y_vals, x_vals = np.mgrid[y_min:y_max, x_min:x_max]
                    p0 = [score, cx - x_min, cy - y_min, 1.0, 1.0, 0.0]
                    p_opt, _ = opt.leastsq(error, p0, args=(x_vals, y_vals, local_region))
                    refined_x = p_opt[1] + x_min
                    refined_y = p_opt[2] + y_min
                    refined_centers.append((float(refined_x), float(refined_y)))
                except:
                    refined_centers.append((float(cx), float(cy)))
            else:
                refined_centers.append((float(cx), float(cy)))
        return refined_centers, scores

class NMS:
    def __init__(self,
                 nms_threshold: float = 0.3,
                 min_particle_size: int = 32,
                 max_particle_size: int = 256,
                 score_threshold: float = 0.5):
        self.nms_threshold = nms_threshold
        self.min_particle_size = min_particle_size
        self.max_particle_size = max_particle_size
        self.score_threshold = score_threshold
        logger.info(f"NMS initialized: threshold={nms_threshold}, "
                    f"particle_size_range=[{min_particle_size}, {max_particle_size}]")

    def _generate_bboxes(self,
                         centers: List[Tuple[int, int]],
                         scores: List[float],
                         image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        bboxes = []
        for (cx, cy), score in zip(centers, scores):
            if score < self.score_threshold:
                continue
            size = self.min_particle_size + min(
                self.max_particle_size - self.min_particle_size,
                int(score * (self.max_particle_size - self.min_particle_size))
            )
            half_size = size // 2
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w - 1, cx + half_size)
            y2 = min(h - 1, cy + half_size)
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2, score, cx, cy])
        return np.array(bboxes) if bboxes else np.empty((0, 7))

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1, y1, x2, y2 = box1[:4]
        x1p, y1p, x2p, y2p = box2[:4]
        xi1, yi1 = max(x1, x1p), max(y1, y1p)
        xi2, yi2 = min(x2, x2p), min(y2, y2p)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2p - x1p) * (y2p - y1p)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def __call__(self,
                 centers: List[Tuple[int, int]],
                 scores: List[float],
                 image_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[float]]:
        if len(centers) == 0:
            return [], []
        bboxes = self._generate_bboxes(centers, scores, image_shape)
        if bboxes.shape[0] == 0:
            return [], []
        order = np.argsort(bboxes[:, 4])[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = np.array([self._compute_iou(bboxes[i], bboxes[j]) for j in order[1:]])
            order = order[1:][ious <= self.nms_threshold]
        kept_bboxes = bboxes[keep]
        final_centers = [(int(cx), int(cy)) for cx, cy in zip(kept_bboxes[:, 5], kept_bboxes[:, 6])]
        final_scores = [float(s) for s in kept_bboxes[:, 4]]
        return final_centers, final_scores

def extract_coordinates(segmentation_mask: np.ndarray,
                        min_area: int = 10,
                        max_area: Optional[int] = None) -> Tuple[List[Tuple[int, int]], List[float], List[int]]:
    if segmentation_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {segmentation_mask.shape}")
    labeled, num_regions = ndimage.label(segmentation_mask.astype(np.int32))
    if num_regions == 0:
        return [], [], []
    properties = ndimage.measurements.center_of_mass(
        segmentation_mask, labeled, index=range(1, num_regions + 1)
    )
    areas = ndimage.sum(segmentation_mask, labeled, index=range(1, num_regions + 1))
    max_probs = ndimage.maximum(segmentation_mask, labeled, index=range(1, num_regions + 1))
    centers = []
    scores = []
    valid_areas = []
    for i, (prop, area, max_prob) in enumerate(zip(properties, areas, max_probs)):
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        cy, cx = prop
        centers.append((int(cx), int(cy)))
        scores.append(float(max_prob))
        valid_areas.append(int(area))
    return centers, scores, valid_areas

def filter_by_size(coordinates: List[Tuple[int, int]],
                   scores: List[float],
                   prob_map: np.ndarray,
                   min_radius: int = 16,
                   max_radius: int = 128) -> Tuple[List[Tuple[int, int]], List[float]]:
    filtered_coords = []
    filtered_scores = []
    for (cx, cy), score in zip(coordinates, scores):
        radius = min_radius + int(score * (max_radius - min_radius))
        y_min, y_max = max(0, cy - radius), min(prob_map.shape[0], cy + radius + 1)
        x_min, x_max = max(0, cx - radius), min(prob_map.shape[1], cx + radius + 1)
        local_region = prob_map[y_min:y_max, x_min:x_max]
        if local_region.size == 0:
            continue
        local_mask = local_region > 0.5
        if np.sum(local_mask) < 10:
            continue
        filtered_coords.append((cx, cy))
        filtered_scores.append(score)
    return filtered_coords, filtered_scores

def _fit_gaussian_2d(data: np.ndarray) -> Optional[Dict[str, float]]:
    if data.size < 9:
        return None
    try:
        h, w = data.shape
        y, x = np.mgrid[0:h, 0:w]
        x = x.flatten()
        y = y.flatten()
        z = data.flatten()
        A = np.column_stack([x*x, y*y, x*y, x, y, np.ones_like(x)])
        coeffs, _, _, _ = np.linalg.lstsq(A, np.log(np.clip(z, 1e-10, None)), rcond=None)
        a, b, c, d, e, f = coeffs
        denom = 4*a*b - c*c
        if abs(denom) < 1e-10:
            return None
        x0 = (c*e - 2*b*d) / denom
        y0 = (c*d - 2*a*e) / denom
        sigma_x = np.sqrt(1.0 / abs(2*a)) if abs(a) > 1e-10 else 1.0
        sigma_y = np.sqrt(1.0 / abs(2*b)) if abs(b) > 1e-10 else 1.0
        amplitude = np.exp(f - (a*x0*x0 + b*y0*y0 + c*x0*y0))
        return {
            'x0': x0, 'y0': y0,
            'sigma_x': sigma_x, 'sigma_y': sigma_y,
            'amplitude': amplitude
        }
    except Exception:
        return None

def detect_with_subpixel(prob_map: np.ndarray,
                         threshold: float = 0.5,
                         min_distance: int = 20,
                         window_size: int = 7) -> List[Dict[str, Any]]:
    detector = PeakDetector(
        threshold_abs=threshold,
        min_distance=min_distance
    )
    peaks, scores = detector.detect(prob_map)
    nms = NMS(score_threshold=threshold)
    final_centers, final_scores = nms(peaks, scores, prob_map.shape)
    nms_peaks = [(x, y, s) for (x, y), s in zip(final_centers, final_scores)]
    results = []
    half_win = window_size // 2
    for x, y, score in nms_peaks:
        y_min = max(0, y - half_win)
        y_max = min(prob_map.shape[0], y + half_win + 1)
        x_min = max(0, x - half_win)
        x_max = min(prob_map.shape[1], x + half_win + 1)
        local_region = prob_map[y_min:y_max, x_min:x_max]
        if local_region.size < 9:
            results.append({
                'x': float(x), 'y': float(y),
                'score': float(score),
                'radius': float(min_distance),
                'snr': 0.0
            })
            continue
        fit = _fit_gaussian_2d(local_region)
        if fit is not None:
            offset_y = y - half_win
            offset_x = x - half_win
            sub_x = offset_x + fit['x0']
            sub_y = offset_y + fit['y0']
            radius = (fit['sigma_x'] + fit['sigma_y']) * 0.5
            if radius < 5:
                radius = min_distance / 2
            local_max = float(local_region.max())
            local_background = float(np.percentile(local_region, 10))
            snr = (local_max - local_background) / max(local_region.std(), 1e-6)
            results.append({
                'x': float(sub_x), 'y': float(sub_y),
                'score': float(score),
                'radius': float(radius),
                'snr': float(snr)
            })
        else:
            results.append({
                'x': float(x), 'y': float(y),
                'score': float(score),
                'radius': float(min_distance / 2),
                'snr': 0.0
            })
    return results
