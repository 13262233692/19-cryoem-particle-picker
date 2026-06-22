import numpy as np
from scipy import ndimage, optimize
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from src.utils.logging import get_logger

logger = get_logger("postprocessing.refinement")

@dataclass
class RefinedParticle:
    x: float
    y: float
    score: float
    radius: float
    eccentricity: float = 0.0
    orientation: float = 0.0
    snr: float = 0.0

class ParticleRefiner:
    def __init__(self,
                 initial_radius: int = 32,
                 fit_radius_range: Tuple[int, int] = (16, 128),
                 subpixel: bool = True,
                 max_iterations: int = 50):
        self.initial_radius = initial_radius
        self.fit_radius_range = fit_radius_range
        self.subpixel = subpixel
        self.max_iterations = max_iterations
        logger.info(f"ParticleRefiner initialized: radius_range={fit_radius_range}, "
                    f"subpixel={subpixel}")

    def _gaussian2d(self, p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        A, x0, y0, sigma_x, sigma_y, theta, bg = p
        a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
        c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
        return bg + A * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))

    def _error_function(self, p: np.ndarray, x: np.ndarray, y: np.ndarray,
                        z: np.ndarray) -> np.ndarray:
        return np.ravel(self._gaussian2d(p, x, y) - z)

    def fit_gaussian(self, image: np.ndarray, cx: int, cy: int,
                     window_size: Optional[int] = None) -> Optional[RefinedParticle]:
        if window_size is None:
            window_size = self.initial_radius * 2
        half_w = window_size // 2
        y_min, y_max = max(0, cy - half_w), min(image.shape[0], cy + half_w)
        x_min, x_max = max(0, cx - half_w), min(image.shape[1], cx + half_w)
        if (y_max - y_min) < 5 or (x_max - x_min) < 5:
            return None
        local_region = image[y_min:y_max, x_min:x_max].astype(np.float64)
        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
        A0 = local_region.max() - local_region.min()
        bg0 = local_region.min()
        p0 = [A0, cx, cy, 10.0, 10.0, 0.0, bg0]
        bounds = (
            [0.0, x_min, y_min, 2.0, 2.0, -np.pi/2, -np.inf],
            [np.inf, x_max, y_max, half_w, half_w, np.pi/2, np.inf]
        )
        try:
            result = optimize.least_squares(
                self._error_function, p0,
                args=(x_grid, y_grid, local_region),
                bounds=bounds,
                max_nfev=self.max_iterations
            )
            p_opt = result.x
            A, x0, y0, sigma_x, sigma_y, theta, bg = p_opt
            radius = max(sigma_x, sigma_y) * 2.355
            eccentricity = 1.0 - min(sigma_x, sigma_y) / max(sigma_x, sigma_y) if max(sigma_x, sigma_y) > 0 else 0
            signal = A
            noise = np.std(local_region - self._gaussian2d(p_opt, x_grid, y_grid))
            snr = signal / noise if noise > 0 else 0
            score = min(1.0, A / (A + bg))
            return RefinedParticle(
                x=float(x0),
                y=float(y0),
                score=float(score),
                radius=float(radius),
                eccentricity=float(eccentricity),
                orientation=float(theta),
                snr=float(snr)
            )
        except Exception as e:
            logger.debug(f"Gaussian fitting failed: {e}")
            return None

    def refine_coordinates(self, image: np.ndarray,
                          coordinates: List[Tuple[int, int]],
                          scores: Optional[List[float]] = None) -> List[RefinedParticle]:
        if scores is None:
            scores = [1.0] * len(coordinates)
        refined = []
        for (cx, cy), score in zip(coordinates, scores):
            radius = int(self.initial_radius * (0.5 + score * 0.5))
            particle = self.fit_gaussian(image, cx, cy, window_size=radius * 2)
            if particle is not None:
                if score < 1.0:
                    particle.score = float(particle.score * 0.7 + score * 0.3)
                refined.append(particle)
        return refined

    def filter_particles(self, particles: List[RefinedParticle],
                         min_score: float = 0.3,
                         min_snr: float = 0.5,
                         max_eccentricity: float = 0.7,
                         radius_range: Optional[Tuple[float, float]] = None) -> List[RefinedParticle]:
        filtered = []
        for p in particles:
            if p.score < min_score:
                continue
            if p.snr < min_snr:
                continue
            if p.eccentricity > max_eccentricity:
                continue
            if radius_range is not None:
                if p.radius < radius_range[0] or p.radius > radius_range[1]:
                    continue
            filtered.append(p)
        return filtered

def remove_duplicates(particles: List[RefinedParticle],
                      min_distance: float = 20.0) -> List[RefinedParticle]:
    if len(particles) == 0:
        return []
    sorted_idx = np.argsort([p.score for p in particles])[::-1]
    sorted_particles = [particles[i] for i in sorted_idx]
    keep = [True] * len(sorted_particles)
    for i in range(len(sorted_particles)):
        if not keep[i]:
            continue
        p1 = sorted_particles[i]
        for j in range(i + 1, len(sorted_particles)):
            if not keep[j]:
                continue
            p2 = sorted_particles[j]
            dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
            if dist < min_distance:
                keep[j] = False
    return [sorted_particles[i] for i in range(len(sorted_particles)) if keep[i]]

def compute_class_averages(image: np.ndarray,
                           particles: List[RefinedParticle],
                           box_size: int = 64,
                           normalize: bool = True) -> Tuple[np.ndarray, List[RefinedParticle]]:
    if len(particles) == 0:
        return np.empty((0, box_size, box_size)), []
    half_box = box_size // 2
    particle_images = []
    valid_particles = []
    for p in particles:
        x, y = int(round(p.x)), int(round(p.y))
        y_min, y_max = y - half_box, y + half_box
        x_min, x_max = x - half_box, x + half_box
        if (y_min < 0 or y_max > image.shape[0] or
            x_min < 0 or x_max > image.shape[1]):
            continue
        particle_img = image[y_min:y_max, x_min:x_max].astype(np.float32)
        if normalize:
            mean = np.mean(particle_img)
            std = np.std(particle_img) + 1e-8
            particle_img = (particle_img - mean) / std
        particle_images.append(particle_img)
        valid_particles.append(p)
    if not particle_images:
        return np.empty((0, box_size, box_size)), []
    return np.stack(particle_images, axis=0), valid_particles

def export_coordinates(particles: List[RefinedParticle],
                       output_path: str,
                       format: str = "star") -> str:
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if format == "star":
        with open(output_path, "w") as f:
            f.write("\n# version 30001\n\ndata_\n\n")
            f.write("loop_\n")
            f.write("_rlnCoordinateX #1\n")
            f.write("_rlnCoordinateY #2\n")
            f.write("_rlnParticleScore #3\n")
            f.write("_rlnParticleRadius #4\n")
            f.write("_rlnParticleSNR #5\n")
            for p in particles:
                f.write(f"{p.x:.4f} {p.y:.4f} {p.score:.4f} {p.radius:.4f} {p.snr:.4f}\n")
    elif format == "csv":
        with open(output_path, "w") as f:
            f.write("x,y,score,radius,snr,eccentricity,orientation\n")
            for p in particles:
                f.write(f"{p.x:.4f},{p.y:.4f},{p.score:.4f},{p.radius:.4f},"
                       f"{p.snr:.4f},{p.eccentricity:.4f},{p.orientation:.4f}\n")
    elif format == "tsv":
        with open(output_path, "w") as f:
            f.write("x\ty\tscore\tradius\tsnr\teccentricity\torientation\n")
            for p in particles:
                f.write(f"{p.x:.4f}\t{p.y:.4f}\t{p.score:.4f}\t{p.radius:.4f}\t"
                       f"{p.snr:.4f}\t{p.eccentricity:.4f}\t{p.orientation:.4f}\n")
    elif format == "coords":
        with open(output_path, "w") as f:
            for p in particles:
                f.write(f"{p.x:.4f}\t{p.y:.4f}\n")
    else:
        np.save(output_path, np.array([[p.x, p.y, p.score, p.radius, p.snr] for p in particles]))
    logger.info(f"Exported {len(particles)} coordinates to {output_path} ({format})")
    return output_path
