"""
3D Orientation Estimation based on Fourier Slice Theorem.

Implements unsupervised 3D orientation estimation using the classical
Fourier Slice Theorem (Central Slice Theorem) with common-line intersection
optimization and conjugate gradient refinement.

Key concepts:
- Fourier Slice Theorem: The 2D Fourier transform of a projection of a 3D
  object equals a central slice of the 3D Fourier transform of that object,
  passing through the origin and oriented perpendicular to the projection
  direction.
- Common Line: Any two central slices (from projections at different
  orientations) intersect along a line passing through the origin - this is
  their common line. The orientation difference can be recovered from the
  angular relationship of these common lines.
- Conjugate Gradient: Used to refine orientation estimates by minimizing
  the common-line residual error in reciprocal space.
"""

import os
import gc
import weakref
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
from src.utils.logging import get_logger

logger = get_logger("reconstruction.orientation")

PHYSICAL_CRITICAL_RESIDUAL = 0.75
DEFAULT_CRITICAL_ANGULAR_SPREAD = 15.0


@dataclass
class EulerAngles:
    phi: float
    theta: float
    psi: float

    def to_rotation_matrix(self) -> np.ndarray:
        r = R.from_euler('ZYZ', [self.phi, self.theta, self.psi], degrees=True)
        return r.as_matrix()

    @classmethod
    def from_rotation_matrix(cls, mat: np.ndarray) -> "EulerAngles":
        r = R.from_matrix(mat)
        phi, theta, psi = r.as_euler('ZYZ', degrees=True)
        return cls(phi=float(phi), theta=float(theta), psi=float(psi))

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.phi, self.theta, self.psi)


@dataclass
class ProjectionSpectrum:
    particle_id: int
    image: np.ndarray
    fft2: Optional[np.ndarray] = None
    orientation: Optional[EulerAngles] = None
    orientation_confidence: float = 0.0

    def ensure_fft(self) -> np.ndarray:
        if self.fft2 is None:
            self.fft2 = np.fft.fftshift(np.fft.fft2(np.asarray(self.image, dtype=np.float32)))
        return self.fft2

    def cleanup(self) -> None:
        if self.fft2 is not None:
            del self.fft2
            self.fft2 = None
        if hasattr(self, 'image') and self.image is not None:
            self.image = None
        gc.collect()


@dataclass
class CommonLine:
    idx_a: int
    idx_b: int
    angle_a: float
    angle_b: float
    residual: float


@dataclass
class OrientationResult:
    orientations: List[EulerAngles]
    residuals: np.ndarray
    mean_residual: float
    median_residual: float
    max_residual: float
    min_residual: float
    angular_spread_deg: float
    orientation_coverage: float
    is_preferred_orientation: bool
    euler_histogram: Dict[str, Any]
    common_lines: List[CommonLine]
    processing_time_ms: float
    particle_count: int


class FourierSliceEstimator:
    """
    3D orientation estimator using the Fourier Slice Theorem.

    Workflow:
    1. Compute 2D FFT for each projection (central slices in reciprocal space)
    2. Find common lines between every pair of central slices
    3. Build a system of equations from common-line angular relationships
    4. Solve initial orientation estimate via eigen-decomposition
    5. Refine using non-linear conjugate gradient to minimize common-line residual

    Usage:
        estimator = FourierSliceEstimator()
        spectra = [ProjectionSpectrum(id, img) for id, img in projections]
        result = estimator.estimate_batch(spectra)
        if result.is_preferred_orientation:
            # block high-resolution refinement for this batch
    """

    def __init__(self,
                 critical_residual_threshold: float = PHYSICAL_CRITICAL_RESIDUAL,
                 critical_angular_spread_deg: float = DEFAULT_CRITICAL_ANGULAR_SPREAD,
                 n_ray_samples: int = 180,
                 max_cg_iterations: int = 50,
                 cg_tolerance: float = 1e-4,
                 fft_pad_factor: int = 2):
        self.critical_residual_threshold = critical_residual_threshold
        self.critical_angular_spread_deg = critical_angular_spread_deg
        self.n_ray_samples = n_ray_samples
        self.max_cg_iterations = max_cg_iterations
        self.cg_tolerance = cg_tolerance
        self.fft_pad_factor = fft_pad_factor
        self._weak_refs: List[weakref.ReferenceType] = []
        logger.info(
            f"FourierSliceEstimator initialized: "
            f"critical_residual={critical_residual_threshold:.3f}, "
            f"critical_spread={critical_angular_spread_deg:.1f}deg, "
            f"n_rays={n_ray_samples}, max_cg_iter={max_cg_iterations}"
        )

    def _register_ref(self, obj) -> None:
        try:
            self._weak_refs.append(weakref.ref(obj))
        except Exception:
            pass

    def _purge_refs(self) -> None:
        try:
            self._weak_refs = [r for r in self._weak_refs if r() is not None]
        except Exception:
            pass

    def _extract_polar_spectrum(self, fft2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = fft2d.shape
        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)
        angles = np.linspace(0, np.pi, self.n_ray_samples, endpoint=False)
        rays = np.zeros((self.n_ray_samples, max_r), dtype=np.float32)
        thetas = np.linspace(0, 2 * np.pi, max_r, endpoint=False)
        radii = np.arange(max_r)
        for i, angle in enumerate(angles):
            xs = cx + (radii * np.cos(angle)).astype(np.int32)
            ys = cy + (radii * np.sin(angle)).astype(np.int32)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            rays[i] = np.abs(fft2d[ys, xs])
        return rays, angles

    def _find_common_line(self,
                          rays_a: np.ndarray, rays_b: np.ndarray,
                          angles: np.ndarray) -> CommonLine:
        n_angles = len(angles)
        best_residual = np.inf
        best_ang_a = 0.0
        best_ang_b = 0.0
        rays_a_norm = rays_a / (np.linalg.norm(rays_a, axis=1, keepdims=True) + 1e-12)
        rays_b_norm = rays_b / (np.linalg.norm(rays_b, axis=1, keepdims=True) + 1e-12)
        for i in range(n_angles):
            corr = rays_b_norm @ rays_a_norm[i]
            j = int(np.argmax(corr))
            resid = 1.0 - float(corr[j])
            if resid < best_residual:
                best_residual = resid
                best_ang_a = float(angles[i])
                best_ang_b = float(angles[j])
        return CommonLine(
            idx_a=-1, idx_b=-1,
            angle_a=best_ang_a,
            angle_b=best_ang_b,
            residual=best_residual
        )

    def _build_common_line_matrix(self,
                                  spectra: List[ProjectionSpectrum]) -> Tuple[np.ndarray, List[CommonLine]]:
        n = len(spectra)
        common_lines: List[CommonLine] = []
        all_rays = []
        all_angles = None
        for spec in spectra:
            fft = spec.ensure_fft()
            rays, angles = self._extract_polar_spectrum(fft)
            all_rays.append(rays)
            if all_angles is None:
                all_angles = angles
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                cl = self._find_common_line(all_rays[i], all_rays[j], all_angles)
                cl.idx_a = i
                cl.idx_b = j
                common_lines.append(cl)
                delta = cl.angle_b - cl.angle_a
                A[i, j] = np.cos(delta)
                A[j, i] = np.cos(delta)
                A[i, i] = 1.0
                A[j, j] = 1.0
        del all_rays
        return A, common_lines

    def _initial_orientation_eigen(self, A: np.ndarray) -> List[EulerAngles]:
        n = A.shape[0]
        eigvals, eigvecs = np.linalg.eigh(A)
        order = np.argsort(-eigvals)
        top = eigvecs[:, order[:3]]
        orientations = []
        for i in range(n):
            v = top[i]
            v = v / (np.linalg.norm(v) + 1e-12)
            theta = float(np.arccos(np.clip(v[2], -1.0, 1.0)) * 180.0 / np.pi)
            phi = float(np.arctan2(v[1], v[0]) * 180.0 / np.pi)
            psi = 0.0
            orientations.append(EulerAngles(phi=phi, theta=theta, psi=psi))
        return orientations

    def _common_line_residual(self, orientations: List[EulerAngles],
                              common_lines: List[CommonLine]) -> np.ndarray:
        n = len(orientations)
        residuals = np.zeros(len(common_lines), dtype=np.float32)
        for idx, cl in enumerate(common_lines):
            rot_a = orientations[cl.idx_a].to_rotation_matrix()
            rot_b = orientations[cl.idx_b].to_rotation_matrix()
            line_a = rot_a @ np.array([np.cos(cl.angle_a), np.sin(cl.angle_a), 0.0])
            line_b = rot_b @ np.array([np.cos(cl.angle_b), np.sin(cl.angle_b), 0.0])
            line_a = line_a / (np.linalg.norm(line_a) + 1e-12)
            line_b = line_b / (np.linalg.norm(line_b) + 1e-12)
            residuals[idx] = 1.0 - float(np.dot(line_a, line_b))
        return residuals

    def _conjugate_gradient_refine(self,
                                    orientations: List[EulerAngles],
                                    common_lines: List[CommonLine]) -> List[EulerAngles]:
        n = len(orientations)
        params = np.zeros((n, 3), dtype=np.float64)
        for i, o in enumerate(orientations):
            params[i] = [o.phi, o.theta, o.psi]
        def residual_fn(p):
            ors = [EulerAngles(phi=float(p[i, 0]), theta=float(p[i, 1]), psi=float(p[i, 2])) for i in range(n)]
            res = self._common_line_residual(ors, common_lines)
            return float(np.mean(res)), ors

        prev_loss, _ = residual_fn(params)
        velocity = np.zeros_like(params)
        for it in range(self.max_cg_iterations):
            eps = 1e-3
            grad = np.zeros_like(params)
            for i in range(n):
                for k in range(3):
                    params_p = params.copy()
                    params_m = params.copy()
                    params_p[i, k] += eps
                    params_m[i, k] -= eps
                    loss_p, _ = residual_fn(params_p)
                    loss_m, _ = residual_fn(params_m)
                    grad[i, k] = (loss_p - loss_m) / (2 * eps)
            if it == 0:
                velocity = -grad
            else:
                beta_num = np.sum(grad * (grad - velocity))
                beta_den = np.sum(velocity * velocity) + 1e-12
                beta = max(0.0, beta_num / beta_den)
                velocity = -grad + beta * velocity
            step = 1e-2
            best_step_params = params + step * velocity
            best_loss, _ = residual_fn(best_step_params)
            for trial in [1e-1, 1e-2, 1e-3, 1e-4]:
                trial_params = params + trial * velocity
                trial_loss, _ = residual_fn(trial_params)
                if trial_loss < best_loss:
                    best_loss = trial_loss
                    best_step_params = trial_params
            params = best_step_params
            cur_loss = best_loss
            if abs(prev_loss - cur_loss) < self.cg_tolerance * prev_loss:
                logger.debug(f"CG converged at iter {it}: loss={cur_loss:.6f}")
                break
            prev_loss = cur_loss
            if (it + 1) % 10 == 0:
                logger.debug(f"CG iter {it+1}/{self.max_cg_iterations}: loss={cur_loss:.6f}")
        refined = []
        for i in range(n):
            refined.append(EulerAngles(
                phi=float(params[i, 0]),
                theta=float(params[i, 1]),
                psi=float(params[i, 2])
            ))
        return refined

    def _compute_angular_spread(self, orientations: List[EulerAngles]) -> Tuple[float, float]:
        n = len(orientations)
        if n < 2:
            return 0.0, 0.0
        rotvecs = []
        for o in orientations:
            r = R.from_euler('ZYZ', [o.phi, o.theta, o.psi], degrees=True)
            rotvecs.append(r.as_rotvec())
        rotvecs = np.array(rotvecs)
        mean_vec = np.mean(rotvecs, axis=0)
        deviations = []
        for rv in rotvecs:
            diff = np.linalg.norm(rv - mean_vec)
            deviations.append(np.degrees(diff))
        spread = float(np.std(deviations))
        thetas = np.array([o.theta for o in orientations])
        phis = np.array([o.phi for o in orientations])
        theta_coverage = (np.percentile(thetas, 90) - np.percentile(thetas, 10)) / 180.0
        phi_range = (np.percentile(phis, 90) - np.percentile(phis, 10)) / 360.0
        coverage = float(max(theta_coverage, phi_range))
        return spread, coverage

    def _compute_euler_histogram(self, orientations: List[EulerAngles]) -> Dict[str, Any]:
        phis = [o.phi for o in orientations]
        thetas = [o.theta for o in orientations]
        psis = [o.psi for o in orientations]
        phi_edges = np.linspace(-180, 180, 37)
        theta_edges = np.linspace(0, 180, 19)
        psi_edges = np.linspace(-180, 180, 37)
        phi_hist, _ = np.histogram(phis, bins=phi_edges)
        theta_hist, _ = np.histogram(thetas, bins=theta_edges)
        psi_hist, _ = np.histogram(psis, bins=psi_edges)
        return {
            "phi_bins": phi_edges[:-1].tolist(),
            "phi_counts": phi_hist.tolist(),
            "theta_bins": theta_edges[:-1].tolist(),
            "theta_counts": theta_hist.tolist(),
            "psi_bins": psi_edges[:-1].tolist(),
            "psi_counts": psi_hist.tolist(),
            "n_orientations": len(orientations)
        }

    def estimate_batch(self, spectra: List[ProjectionSpectrum]) -> OrientationResult:
        import time
        start = time.perf_counter()
        n = len(spectra)
        if n < 3:
            logger.warning(f"Need at least 3 projections for orientation estimation, got {n}")
            raise ValueError(f"Need at least 3 projections, got {n}")
        logger.info(f"Estimating orientations for {n} projections via Fourier Slice Theorem...")
        A, common_lines = self._build_common_line_matrix(spectra)
        initial_orients = self._initial_orientation_eigen(A)
        refined = self._conjugate_gradient_refine(initial_orients, common_lines)
        residuals = self._common_line_residual(refined, common_lines)
        spread, coverage = self._compute_angular_spread(refined)
        mean_res = float(np.mean(residuals))
        median_res = float(np.median(residuals))
        max_res = float(np.max(residuals))
        min_res = float(np.min(residuals))
        is_preferred = (
            mean_res > self.critical_residual_threshold
            or spread < self.critical_angular_spread_deg
            or coverage < 0.2
        )
        euler_hist = self._compute_euler_histogram(refined)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = OrientationResult(
            orientations=refined,
            residuals=residuals,
            mean_residual=mean_res,
            median_residual=median_res,
            max_residual=max_res,
            min_residual=min_res,
            angular_spread_deg=spread,
            orientation_coverage=coverage,
            is_preferred_orientation=is_preferred,
            euler_histogram=euler_hist,
            common_lines=common_lines,
            processing_time_ms=elapsed_ms,
            particle_count=n
        )
        logger.info(
            f"Orientation estimation complete: "
            f"n={n}, mean_residual={mean_res:.4f}, "
            f"spread={spread:.2f}deg, coverage={coverage:.3f}, "
            f"preferred_orientation={is_preferred}, "
            f"time={elapsed_ms:.1f}ms"
        )
        if is_preferred:
            logger.warning(
                "PREFERRED ORIENTATION DETECTED — "
                f"mean_residual={mean_res:.4f} (>{self.critical_residual_threshold}) "
                f"OR angular_spread={spread:.1f}deg (<{self.critical_angular_spread_deg}deg) "
                f"OR coverage={coverage:.3f} (<0.2). "
                "Air-water interface adsorption likely present. "
                "Batch should be BLOCKED from high-resolution 3D refinement."
            )
        for spec in spectra:
            self._register_ref(spec)
        self._purge_refs()
        gc.collect()
        return result

    def estimate_batch_from_images(self,
                                   particle_ids: List[int],
                                   images: List[np.ndarray]) -> OrientationResult:
        spectra = [
            ProjectionSpectrum(particle_id=int(pid), image=np.asarray(img, dtype=np.float32))
            for pid, img in zip(particle_ids, images)
        ]
        try:
            return self.estimate_batch(spectra)
        finally:
            for s in spectra:
                s.cleanup()
            del spectra
            gc.collect()

    def close(self) -> None:
        try:
            for ref in self._weak_refs:
                try:
                    obj = ref()
                    if obj is not None and hasattr(obj, 'cleanup'):
                        obj.cleanup()
                except Exception:
                    pass
            self._weak_refs.clear()
        except Exception:
            pass
        gc.collect()
        logger.info("FourierSliceEstimator closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
