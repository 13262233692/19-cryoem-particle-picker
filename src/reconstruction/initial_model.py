"""
3D Initial Model Conjugate Gradient Refinement & Particle Batch Cache Manager.

Implements:
1. Back-projection based 3D initial model reconstruction from 2D projections
2. Non-linear conjugate gradient refinement of the 3D density map
3. Projection slice bypass caching for real-time analysis pipeline integration
4. Preferred orientation batch interception with event-driven notification
"""

import os
import gc
import time
import weakref
import threading
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from src.utils.logging import get_logger
from .orientation import (
    FourierSliceEstimator,
    OrientationResult,
    ProjectionSpectrum,
    EulerAngles,
    PHYSICAL_CRITICAL_RESIDUAL,
)

logger = get_logger("reconstruction.initial_model")


@dataclass
class ParticleProjection:
    particle_id: int
    task_id: str
    micrograph_id: str
    center_x: int
    center_y: int
    box_size: int
    image: Optional[np.ndarray] = None
    score: float = 0.0
    captured_at: float = field(default_factory=time.time)

    def cleanup(self) -> None:
        if self.image is not None:
            try:
                del self.image
            except Exception:
                pass
            self.image = None


@dataclass
class ParticleBatch:
    batch_id: str
    task_id: str
    particles: List[ParticleProjection]
    orientation_result: Optional[OrientationResult] = None
    is_blocked: bool = False
    block_reason: Optional[str] = None
    analyzed_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)

    def cleanup(self) -> None:
        for p in self.particles:
            p.cleanup()
        if self.orientation_result is not None:
            try:
                del self.orientation_result
            except Exception:
                pass
            self.orientation_result = None

    @property
    def particle_count(self) -> int:
        return len(self.particles)


class InitialModelRefiner:
    """
    Conjugate-gradient based 3D initial model refinement.

    Uses simple back-projection to create an initial volume, then refines
    by minimizing the projection-matching residual via conjugate gradient.
    """

    def __init__(self,
                 volume_size: int = 128,
                 max_cg_iterations: int = 30,
                 cg_tolerance: float = 1e-5):
        self.volume_size = volume_size
        self.max_cg_iterations = max_cg_iterations
        self.cg_tolerance = cg_tolerance
        self._volume: Optional[np.ndarray] = None
        logger.info(f"InitialModelRefiner initialized: volume_size={volume_size}, "
                   f"max_cg_iter={max_cg_iterations}")

    def _extract_particle_patch(self,
                                 micrograph: np.ndarray,
                                 cx: int, cy: int, box: int) -> np.ndarray:
        h, w = micrograph.shape
        half = box // 2
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        px0, px1 = max(0, -x0), max(0, x1 - w)
        py0, py1 = max(0, -y0), max(0, y1 - h)
        x0c, x1c = max(0, x0), min(w, x1)
        y0c, y1c = max(0, y0), min(h, y1)
        patch = np.zeros((box, box), dtype=np.float32)
        if x1c > x0c and y1c > y0c:
            patch[py0:py0 + (y1c - y0c), px0:px0 + (x1c - x0c)] = micrograph[y0c:y1c, x0c:x1c]
        return patch

    def _back_project(self,
                      projections: List[np.ndarray],
                      orientations: List[EulerAngles]) -> np.ndarray:
        size = self.volume_size
        volume = np.zeros((size, size, size), dtype=np.float32)
        counts = np.zeros((size, size, size), dtype=np.float32)
        center = size // 2
        z_coords, y_coords, x_coords = np.mgrid[
            -center:center + (1 if size % 2 else 0),
            -center:center + (1 if size % 2 else 0),
            -center:center + (1 if size % 2 else 0)
        ].astype(np.float32)
        for idx, (proj, orient) in enumerate(zip(projections, orientations)):
            if idx % 50 == 0:
                logger.debug(f"Back-projecting {idx}/{len(projections)}...")
            rot = orient.to_rotation_matrix().astype(np.float32)
            ph, pw = proj.shape
            pcx, pcy = pw // 2, ph // 2
            proj_3d_x = rot[0, 0] * x_coords + rot[0, 1] * y_coords + rot[0, 2] * z_coords
            proj_3d_y = rot[1, 0] * x_coords + rot[1, 1] * y_coords + rot[1, 2] * z_coords
            ix = np.clip(np.round(proj_3d_x + pcx).astype(np.int32), 0, pw - 1)
            iy = np.clip(np.round(proj_3d_y + pcy).astype(np.int32), 0, ph - 1)
            volume += proj[iy, ix]
            counts += 1.0
        counts[counts == 0] = 1.0
        volume /= counts
        return volume

    def _project_volume(self, volume: np.ndarray, orient: EulerAngles,
                        out_shape: Tuple[int, int]) -> np.ndarray:
        ph, pw = out_shape
        pcx, pcy = pw // 2, ph // 2
        sz, sy, sx = volume.shape
        cz, cy, cx = sz // 2, sy // 2, sx // 2
        rot = orient.to_rotation_matrix().astype(np.float32)
        projection = np.zeros(out_shape, dtype=np.float32)
        counts = np.zeros(out_shape, dtype=np.float32)
        zz, yy, xx = np.mgrid[-cz:cz + 1, -cy:cy + 1, -cx:cx + 1].astype(np.float32)
        proj_x = rot[0, 0] * xx + rot[0, 1] * yy + rot[0, 2] * zz
        proj_y = rot[1, 0] * xx + rot[1, 1] * yy + rot[1, 2] * zz
        ix = np.clip(np.round(proj_x + pcx).astype(np.int32), 0, pw - 1)
        iy = np.clip(np.round(proj_y + pcy).astype(np.int32), 0, ph - 1)
        flat_ix = ix.ravel()
        flat_iy = iy.ravel()
        flat_v = volume.ravel()
        for k in range(len(flat_ix)):
            projection[flat_iy[k], flat_ix[k]] += flat_v[k]
            counts[flat_iy[k], flat_ix[k]] += 1.0
        counts[counts == 0] = 1.0
        return projection / counts

    def refine(self,
               projections: List[np.ndarray],
               orientations: List[EulerAngles]) -> Tuple[np.ndarray, List[float]]:
        n = len(projections)
        logger.info(f"Refining initial 3D model from {n} projections (CG iterations={self.max_cg_iterations})...")
        volume = self._back_project(projections, orientations)
        self._volume = volume.copy()
        residuals = []
        def total_residual(vol):
            res = 0.0
            for i in range(n):
                proj_pred = self._project_volume(vol, orientations[i], projections[i].shape)
                res += float(np.mean((proj_pred - projections[i]) ** 2))
            return res / max(1, n)
        params = volume.astype(np.float64).ravel()
        prev_loss = total_residual(volume)
        residuals.append(prev_loss)
        velocity = np.zeros_like(params)
        for it in range(self.max_cg_iterations):
            eps = 1e-2
            grad = np.zeros_like(params)
            base_loss = prev_loss
            for i in range(0, len(params), max(1, len(params) // 50)):
                params_p = params.copy()
                params_m = params.copy()
                params_p[i] += eps
                params_m[i] -= eps
                loss_p = total_residual(params_p.reshape(volume.shape).astype(np.float32))
                loss_m = total_residual(params_m.reshape(volume.shape).astype(np.float32))
                grad[i] = (loss_p - loss_m) / (2 * eps)
            if it == 0:
                velocity = -grad
            else:
                beta_num = np.sum(grad * (grad - velocity))
                beta_den = np.sum(velocity * velocity) + 1e-12
                beta = max(0.0, beta_num / beta_den)
                velocity = -grad + beta * velocity
            best_loss = base_loss
            best_params = params.copy()
            for step in [1e-3, 1e-4, 1e-5, 1e-6]:
                trial = params + step * velocity
                trial_loss = total_residual(trial.reshape(volume.shape).astype(np.float32))
                if trial_loss < best_loss:
                    best_loss = trial_loss
                    best_params = trial
            params = best_params
            cur_loss = best_loss
            residuals.append(cur_loss)
            logger.debug(f"CG iter {it+1}/{self.max_cg_iterations}: loss={cur_loss:.6f} (prev={prev_loss:.6f})")
            if abs(prev_loss - cur_loss) < self.cg_tolerance * max(1e-6, prev_loss):
                logger.info(f"CG converged at iter {it+1}: loss={cur_loss:.6f}")
                break
            prev_loss = cur_loss
        final_vol = params.reshape(volume.shape).astype(np.float32)
        self._volume = final_vol.copy()
        logger.info(f"Initial model refinement complete: final_loss={residuals[-1]:.6f}")
        gc.collect()
        return final_vol, residuals

    @property
    def volume(self) -> Optional[np.ndarray]:
        return self._volume

    def close(self) -> None:
        if self._volume is not None:
            try:
                del self._volume
            except Exception:
                pass
            self._volume = None
        gc.collect()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ParticleBatchCacheManager:
    """
    Real-time bypass capture cache for AI-picked particle projections.

    Features:
    - Real-time interception of particle coordinates from the AI picker
    - Automatic extraction and caching of 2D projection slices from micrographs
    - Batch-based orientation analysis using FourierSliceEstimator
    - Preferred-orientation detection with event-driven blocking
    - Callback-driven frontend notification channel
    - Bounded memory with automatic eviction of old batches
    """

    def __init__(self,
                 batch_size: int = 500,
                 max_cached_batches: int = 10,
                 default_box_size: int = 256,
                 orientation_estimator: Optional[FourierSliceEstimator] = None):
        self.batch_size = batch_size
        self.max_cached_batches = max_cached_batches
        self.default_box_size = default_box_size
        self._pending_particles: Dict[str, List[ParticleProjection]] = defaultdict(list)
        self._completed_batches: deque = deque(maxlen=max_cached_batches)
        self._lock = threading.Lock()
        self._estimator = orientation_estimator or FourierSliceEstimator()
        self._refiner = InitialModelRefiner()
        self._block_callbacks: List[Callable[[ParticleBatch], None]] = []
        self._weak_refs: List[weakref.ReferenceType] = []
        logger.info(
            f"ParticleBatchCacheManager initialized: "
            f"batch_size={batch_size}, max_batches={max_cached_batches}, "
            f"box_size={default_box_size}"
        )

    def register_block_callback(self, cb: Callable[[ParticleBatch], None]) -> None:
        self._block_callbacks.append(cb)
        logger.info(f"Registered block callback #{len(self._block_callbacks)}")

    def capture_particle(self,
                          task_id: str,
                          particle_id: int,
                          micrograph: Optional[np.ndarray],
                          center_x: int, center_y: int,
                          box_size: Optional[int] = None,
                          micrograph_id: str = "",
                          score: float = 0.0) -> Optional[ParticleBatch]:
        box = box_size or self.default_box_size
        image = None
        if micrograph is not None:
            try:
                half = box // 2
                h, w = micrograph.shape
                x0, x1 = max(0, center_x - half), min(w, center_x + half)
                y0, y1 = max(0, center_y - half), min(h, center_y + half)
                patch = np.zeros((box, box), dtype=np.float32)
                if x1 > x0 and y1 > y0:
                    dy0, dy1 = half - (center_y - y0), half + (y1 - center_y)
                    dx0, dx1 = half - (center_x - x0), half + (x1 - center_x)
                    patch[dy0:dy1, dx0:dx1] = micrograph[y0:y1, x0:x1]
                image = patch
            except Exception as e:
                logger.warning(f"Failed to extract particle patch: {e}")
        proj = ParticleProjection(
            particle_id=particle_id,
            task_id=task_id,
            micrograph_id=micrograph_id,
            center_x=center_x,
            center_y=center_y,
            box_size=box,
            image=image,
            score=score
        )
        batch = None
        with self._lock:
            self._pending_particles[task_id].append(proj)
            self._register_ref(proj)
            if len(self._pending_particles[task_id]) >= self.batch_size:
                batch = self._flush_batch_locked(task_id)
        if batch is not None:
            self._analyze_and_maybe_block(batch)
        return batch

    def _register_ref(self, obj) -> None:
        try:
            self._weak_refs.append(weakref.ref(obj))
        except Exception:
            pass

    def _flush_batch_locked(self, task_id: str) -> ParticleBatch:
        particles = self._pending_particles[task_id]
        batch_id = f"{task_id}_{int(time.time()*1000)}"
        batch = ParticleBatch(
            batch_id=batch_id,
            task_id=task_id,
            particles=list(particles)
        )
        self._pending_particles[task_id] = []
        self._completed_batches.append(batch)
        self._evict_if_needed_locked()
        return batch

    def _evict_if_needed_locked(self) -> None:
        while len(self._completed_batches) > self.max_cached_batches:
            old = self._completed_batches.popleft()
            try:
                old.cleanup()
            except Exception:
                pass

    def _analyze_and_maybe_block(self, batch: ParticleBatch) -> None:
        try:
            images = []
            pids = []
            for p in batch.particles:
                if p.image is not None:
                    images.append(p.image)
                    pids.append(p.particle_id)
            if len(images) < 3:
                logger.warning(f"Batch {batch.batch_id}: only {len(images)} particles with images, skipping analysis")
                batch.analyzed_at = time.time()
                return
            if len(images) > 200:
                idx = np.linspace(0, len(images) - 1, 200).astype(int)
                images = [images[i] for i in idx]
                pids = [pids[i] for i in idx]
            result = self._estimator.estimate_batch_from_images(pids, images)
            batch.orientation_result = result
            batch.analyzed_at = time.time()
            if result.is_preferred_orientation:
                batch.is_blocked = True
                reasons = []
                if result.mean_residual > self._estimator.critical_residual_threshold:
                    reasons.append(
                        f"Common-line residual {result.mean_residual:.4f} > "
                        f"physical threshold {self._estimator.critical_residual_threshold:.4f}"
                    )
                if result.angular_spread_deg < self._estimator.critical_angular_spread_deg:
                    reasons.append(
                        f"Angular spread {result.angular_spread_deg:.1f}deg < "
                        f"critical {self._estimator.critical_angular_spread_deg:.1f}deg"
                    )
                if result.orientation_coverage < 0.2:
                    reasons.append(
                        f"Orientation coverage {result.orientation_coverage:.3f} < 0.2 "
                        f"(air-water interface adsorption suspected)"
                    )
                batch.block_reason = " | ".join(reasons)
                logger.critical(
                    f"BATCH BLOCKED: {batch.batch_id} | "
                    f"task={batch.task_id} | n={result.particle_count} | "
                    f"Reason: {batch.block_reason}"
                )
                for cb in self._block_callbacks:
                    try:
                        cb(batch)
                    except Exception as e:
                        logger.error(f"Block callback failed: {e}")
            else:
                logger.info(
                    f"Batch {batch.batch_id}: orientation analysis OK "
                    f"(residual={result.mean_residual:.4f}, "
                    f"spread={result.angular_spread_deg:.1f}deg, "
                    f"coverage={result.orientation_coverage:.3f})"
                )
        except Exception as e:
            logger.error(f"Orientation analysis failed for batch {batch.batch_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def flush_task(self, task_id: str, force: bool = False) -> Optional[ParticleBatch]:
        with self._lock:
            particles = self._pending_particles.get(task_id, [])
            if not particles or (not force and len(particles) < 3):
                return None
            batch = self._flush_batch_locked(task_id)
        if batch is not None:
            self._analyze_and_maybe_block(batch)
        return batch

    def get_batch(self, batch_id: str) -> Optional[ParticleBatch]:
        with self._lock:
            for b in self._completed_batches:
                if b.batch_id == batch_id:
                    return b
        return None

    def get_task_batches(self, task_id: str) -> List[ParticleBatch]:
        with self._lock:
            return [b for b in self._completed_batches if b.task_id == task_id]

    def get_blocked_batches(self) -> List[ParticleBatch]:
        with self._lock:
            return [b for b in self._completed_batches if b.is_blocked]

    def get_pending_count(self, task_id: str) -> int:
        with self._lock:
            return len(self._pending_particles.get(task_id, []))

    def close(self) -> None:
        try:
            with self._lock:
                for task_id in list(self._pending_particles.keys()):
                    for p in self._pending_particles[task_id]:
                        p.cleanup()
                    self._pending_particles[task_id] = []
                while self._completed_batches:
                    b = self._completed_batches.popleft()
                    b.cleanup()
                for ref in self._weak_refs:
                    try:
                        obj = ref()
                        if obj is not None and hasattr(obj, 'cleanup'):
                            obj.cleanup()
                    except Exception:
                        pass
                self._weak_refs.clear()
                self._block_callbacks.clear()
            self._estimator.close()
            self._refiner.close()
        except Exception:
            pass
        gc.collect()
        logger.info("ParticleBatchCacheManager closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
