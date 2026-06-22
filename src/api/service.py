import os
import uuid
import asyncio
import time
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from fastapi import UploadFile, HTTPException, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from src.utils.logging import get_logger
from src.utils.config import load_config, get_config
from src.utils.visualization import save_result_image
from src.io.mrc_parser import MRCStreamParser
from src.io.stream_ops import write_mrc_file
from src.inference.pipeline import InferencePipeline, InferenceResult
from src.postprocessing.refinement import ParticleRefiner, RefinedParticle, export_coordinates, remove_duplicates
from src.api.schemas import (
    TaskStatus, PickingResult, Particle, ImageInfo,
    ProcessingTimes, TaskStatusResponse, PickingRequest,
    EulerHistogram, OrientationAnalysis, BatchStatus,
    TaskOrientationSummary, PreferredOrientationAlert
)

logger = get_logger("api.service")

class PickingService:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.tasks: Dict[str, PickingResult] = {}
        self.inference_pipeline: Optional[InferencePipeline] = None
        self.particle_refiner = ParticleRefiner()
        self.upload_dir = self.config["api"]["upload_dir"]
        self.result_dir = self.config["api"]["result_dir"]
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._model_loaded = False
        self._start_time = time.time()
        from src.reconstruction import ParticleBatchCacheManager
        self.batch_manager = ParticleBatchCacheManager()
        self._pending_alerts: Dict[str, PreferredOrientationAlert] = {}
        self.batch_manager.register_block_callback(self._on_batch_blocked)
        logger.info("PickingService initialized")

    def _on_batch_blocked(self, batch) -> None:
        try:
            from datetime import datetime
            import uuid
            if batch.orientation_result is None:
                return
            euler = batch.orientation_result.euler_histogram or {}
            hist = EulerHistogram(
                phi_bins=list(euler.get("phi_bins", [])),
                phi_counts=list(euler.get("phi_counts", [])),
                theta_bins=list(euler.get("theta_bins", [])),
                theta_counts=list(euler.get("theta_counts", [])),
                psi_bins=list(euler.get("psi_bins", [])),
                psi_counts=list(euler.get("psi_counts", [])),
                n_orientations=euler.get("n_orientations", 0)
            )
            analysis = OrientationAnalysis(
                particle_count=batch.orientation_result.particle_count,
                mean_residual=float(batch.orientation_result.mean_residual),
                median_residual=float(batch.orientation_result.median_residual),
                max_residual=float(batch.orientation_result.max_residual),
                min_residual=float(batch.orientation_result.min_residual),
                angular_spread_deg=float(batch.orientation_result.angular_spread_deg),
                orientation_coverage=float(batch.orientation_result.orientation_coverage),
                is_preferred_orientation=bool(batch.orientation_result.is_preferred_orientation),
                euler_histogram=hist,
                processing_time_ms=float(batch.orientation_result.processing_time_ms),
                common_line_residuals=[float(x) for x in (batch.orientation_result.residuals or [])]
            )
            alert_id = f"alert_{uuid.uuid4().hex[:12]}"
            alert = PreferredOrientationAlert(
                alert_id=alert_id,
                task_id=batch.task_id,
                batch_id=batch.batch_id,
                timestamp=datetime.utcnow(),
                is_preferred_orientation=True,
                block_reason=batch.block_reason or "Unknown preferred orientation",
                particle_count=batch.particle_count,
                orientation=analysis,
                severity="critical",
                action_required=True
            )
            self._pending_alerts[alert_id] = alert
            logger.critical(
                f"Preferred orientation alert generated: "
                f"alert_id={alert_id} task_id={batch.task_id} "
                f"batch_id={batch.batch_id} reason={batch.block_reason}"
            )
        except Exception as e:
            logger.error(f"_on_batch_blocked callback failed: {e}", exc_info=True)

    def get_pending_alerts(self, task_id: Optional[str] = None) -> List[PreferredOrientationAlert]:
        alerts = list(self._pending_alerts.values())
        if task_id:
            alerts = [a for a in alerts if a.task_id == task_id]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def dismiss_alert(self, alert_id: str) -> bool:
        if alert_id in self._pending_alerts:
            del self._pending_alerts[alert_id]
            logger.info(f"Alert {alert_id} dismissed")
            return True
        return False


    def load_model(self, onnx_model_path: Optional[str] = None) -> bool:
        try:
            if onnx_model_path is None:
                onnx_model_path = self.config["inference"]["onnx_model_path"]
            if not os.path.exists(onnx_model_path):
                logger.warning(f"ONNX model not found: {onnx_model_path}")
                self._model_loaded = False
                return False
            self.inference_pipeline = InferencePipeline(
                config_path="configs/config.yaml",
                onnx_model_path=onnx_model_path
            )
            self._model_loaded = True
            logger.info(f"Model loaded successfully from {onnx_model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            return False

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    async def save_uploaded_file(self, file: UploadFile) -> Tuple[str, int]:
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".mrc"
        file_path = os.path.join(self.upload_dir, f"{file_id}{ext}")
        file_size = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                file_size += len(chunk)
        logger.info(f"Saved uploaded file: {file_path}, size={file_size/1024/1024:.2f}MB")
        return file_path, file_size

    def create_task(self, file_name: str, image_info: ImageInfo) -> str:
        task_id = str(uuid.uuid4())
        task = PickingResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            file_name=file_name,
            image_info=image_info,
            num_particles=0,
            particles=[]
        )
        self.tasks[task_id] = task
        logger.info(f"Created task: {task_id} for file: {file_name}")
        return task_id

    def get_task(self, task_id: str) -> Optional[PickingResult]:
        return self.tasks.get(task_id)

    def get_task_status(self, task_id: str) -> TaskStatusResponse:
        task = self.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        progress = 1.0 if task.status == TaskStatus.COMPLETED else 0.5 if task.status == TaskStatus.PROCESSING else 0.0
        return TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            progress=progress,
            message=task.error_message,
            created_at=task.created_at,
            completed_at=task.completed_at
        )

    def _read_mrc_image(self, file_path: str) -> Tuple[np.ndarray, ImageInfo]:
        try:
            with MRCStreamParser(file_path, zero_copy=True) as parser:
                image = parser.get_image(0)
                info = parser.header.to_dict()
                dims = info["dimensions"]
                image_info = ImageInfo(
                    width=dims[0],
                    height=dims[1],
                    pixel_size=info["pixel_size"],
                    data_type=str(parser.dtype),
                    min_value=info["data_range"][0],
                    max_value=info["data_range"][1],
                    mean_value=info["data_mean"]
                )
            return image, image_info
        except Exception as e:
            logger.error(f"Failed to read MRC file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid MRC file: {e}")

    def _convert_to_particles(self,
                               refined_particles: List[RefinedParticle]) -> List[Particle]:
        particles = []
        for p in refined_particles:
            particles.append(Particle(
                x=p.x,
                y=p.y,
                score=p.score,
                radius=p.radius,
                snr=p.snr,
                eccentricity=p.eccentricity,
                orientation=p.orientation
            ))
        return particles

    def process_task_sync(self, task_id: str,
                          file_path: str,
                          request: Optional[PickingRequest] = None) -> None:
        try:
            task = self.tasks.get(task_id)
            if task is None:
                logger.error(f"Task {task_id} not found")
                return
            task.status = TaskStatus.PROCESSING
            self.tasks[task_id] = task
            image, image_info = self._read_mrc_image(file_path)
            if self.inference_pipeline is None:
                if not self.load_model():
                    raise RuntimeError("Model not loaded and failed to load")
            if request is not None:
                if request.confidence_threshold is not None:
                    self.inference_pipeline.confidence_threshold = request.confidence_threshold
                if request.min_distance is not None:
                    self.inference_pipeline.peak_detector.min_distance = request.min_distance
                if request.max_particles is not None:
                    self.inference_pipeline.peak_detector.max_particles = request.max_particles
            result: InferenceResult = self.inference_pipeline.process(image)
            refined = self.particle_refiner.refine_coordinates(
                result.preprocessed_image,
                result.coordinates,
                result.confidence_scores
            )
            refined = self.particle_refiner.filter_particles(refined)
            refined = remove_duplicates(refined)
            for idx, rp in enumerate(refined):
                cx_int = int(round(rp.x))
                cy_int = int(round(rp.y))
                self.batch_manager.capture_particle(
                    task_id=task_id,
                    particle_id=idx,
                    micrograph=result.original_image,
                    center_x=cx_int,
                    center_y=cy_int,
                    box_size=self.particle_refiner.diameter,
                    micrograph_id=task.file_name,
                    score=rp.score
                )
            self.batch_manager.flush_task(task_id, force=True)
            particles = self._convert_to_particles(refined)
            result_dir = os.path.join(self.result_dir, task_id)
            os.makedirs(result_dir, exist_ok=True)
            result_image_path = os.path.join(result_dir, "result.png")
            save_result_image(
                result.original_image,
                result_image_path,
                coordinates=[(int(p.x), int(p.y)) for p in refined],
                heatmap=result.probability_map
            )
            prob_map_path = os.path.join(result_dir, "probability_map.mrc")
            write_mrc_file(prob_map_path, result.probability_map)
            coords_path = None
            if request and request.export_format:
                coords_path = os.path.join(result_dir, f"coordinates.{request.export_format}")
                export_coordinates(refined, coords_path, format=request.export_format)
            else:
                coords_path = os.path.join(result_dir, "coordinates.star")
                export_coordinates(refined, coords_path, format="star")
            processing_times = ProcessingTimes(
                preprocessing=result.processing_time["preprocessing"],
                inference=result.processing_time["inference"],
                postprocessing=result.processing_time["postprocessing"],
                total=result.processing_time["total"]
            )
            task.status = TaskStatus.COMPLETED
            task.image_info = image_info
            task.num_particles = len(particles)
            task.particles = particles
            task.processing_times = processing_times
            task.result_image_url = f"/api/results/{task_id}/image"
            task.probability_map_url = f"/api/results/{task_id}/probability_map"
            task.coordinates_url = f"/api/results/{task_id}/coordinates"
            task.completed_at = datetime.utcnow()
            self.tasks[task_id] = task
            logger.info(f"Task {task_id} completed: {len(particles)} particles detected, "
                       f"total_time={processing_times.total*1000:.2f}ms")
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.completed_at = datetime.utcnow()
                self.tasks[task_id] = task

    async def process_task(self, task_id: str,
                           file_path: str,
                           request: Optional[PickingRequest] = None) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.process_task_sync,
            task_id, file_path, request
        )

    def list_tasks(self, skip: int = 0, limit: int = 100,
                   status: Optional[TaskStatus] = None) -> List[PickingResult]:
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)
        return tasks[skip:skip + limit]

    def export_coordinates(self, task_id: str, format: str = "star") -> str:
        task = self.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        if task.status != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"Task {task_id} not completed")
        result_dir = os.path.join(self.result_dir, task_id)
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, f"coordinates.{format}")
        refined = [
            RefinedParticle(
                x=p.x, y=p.y, score=p.score,
                radius=p.radius or 32.0,
                eccentricity=p.eccentricity or 0.0,
                orientation=p.orientation or 0.0,
                snr=p.snr or 0.0
            ) for p in task.particles
        ]
        return export_coordinates(refined, output_path, format=format)

    def get_result_file_path(self, task_id: str, file_type: str) -> str:
        task = self.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        if task.status != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"Task {task_id} not completed")
        result_dir = os.path.join(self.result_dir, task_id)
        file_map = {
            "image": os.path.join(result_dir, "result.png"),
            "probability_map": os.path.join(result_dir, "probability_map.mrc"),
            "coordinates": os.path.join(result_dir, "coordinates.star"),
        }
        file_path = file_map.get(file_type)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found for task {task_id}")
        return file_path

    def get_system_info(self) -> Dict[str, Any]:
        import psutil
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_memory = None
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory = psutil.virtual_memory()
        return {
            "name": self.config["system"]["name"],
            "version": self.config["system"]["version"],
            "device": self.config["system"]["device"] if gpu_available else "cpu",
            "model_loaded": self._model_loaded,
            "model_path": self.config["inference"]["onnx_model_path"] if self._model_loaded else None,
            "gpu_available": gpu_available,
            "gpu_memory_gb": gpu_memory,
            "total_memory_gb": memory.total / 1024**3,
            "cpu_count": psutil.cpu_count(),
            "uptime": time.time() - self._start_time
        }

    def run_benchmark(self, image_shape: Tuple[int, int] = (4096, 4096),
                      num_runs: int = 10) -> Dict[str, Any]:
        if self.inference_pipeline is None:
            if not self.load_model():
                raise RuntimeError("Model not loaded")
        return self.inference_pipeline.benchmark(image_shape, num_runs)

    def cleanup(self) -> None:
        self.executor.shutdown(wait=True)
        if self.inference_pipeline:
            self.inference_pipeline.close()
        try:
            self.batch_manager.close()
        except Exception:
            pass
        logger.info("PickingService cleaned up")

    def get_task_orientation_summary(self, task_id: str) -> TaskOrientationSummary:
        from datetime import datetime
        batches = self.batch_manager.get_task_batches(task_id)
        pending = self.batch_manager.get_pending_count(task_id)
        total_particles = sum(b.particle_count for b in batches) + pending
        blocked = sum(1 for b in batches if b.is_blocked)
        ok = sum(1 for b in batches if not b.is_blocked and b.analyzed_at is not None)
        statuses = []
        for b in batches:
            orient_data = None
            is_po = None
            if b.orientation_result is not None:
                is_po = b.orientation_result.is_preferred_orientation
                euler = b.orientation_result.euler_histogram or {}
                hist = EulerHistogram(
                    phi_bins=list(euler.get("phi_bins", [])),
                    phi_counts=list(euler.get("phi_counts", [])),
                    theta_bins=list(euler.get("theta_bins", [])),
                    theta_counts=list(euler.get("theta_counts", [])),
                    psi_bins=list(euler.get("psi_bins", [])),
                    psi_counts=list(euler.get("psi_counts", [])),
                    n_orientations=euler.get("n_orientations", 0)
                )
                orient_data = OrientationAnalysis(
                    particle_count=b.orientation_result.particle_count,
                    mean_residual=float(b.orientation_result.mean_residual),
                    median_residual=float(b.orientation_result.median_residual),
                    max_residual=float(b.orientation_result.max_residual),
                    min_residual=float(b.orientation_result.min_residual),
                    angular_spread_deg=float(b.orientation_result.angular_spread_deg),
                    orientation_coverage=float(b.orientation_result.orientation_coverage),
                    is_preferred_orientation=bool(b.orientation_result.is_preferred_orientation),
                    euler_histogram=hist,
                    processing_time_ms=float(b.orientation_result.processing_time_ms),
                    common_line_residuals=[float(x) for x in (b.orientation_result.residuals or [])]
                )
            statuses.append(BatchStatus(
                batch_id=b.batch_id,
                task_id=b.task_id,
                particle_count=b.particle_count,
                is_blocked=b.is_blocked,
                block_reason=b.block_reason,
                analyzed=b.analyzed_at is not None,
                is_preferred_orientation=is_po,
                orientation=orient_data,
                created_at=datetime.utcfromtimestamp(b.created_at),
                analyzed_at=datetime.utcfromtimestamp(b.analyzed_at) if b.analyzed_at else None
            ))
        overall = "blocked" if blocked > 0 else ("processing" if (ok + blocked) < len(batches) else "ok")
        return TaskOrientationSummary(
            task_id=task_id,
            total_particles_captured=total_particles,
            total_batches=len(batches),
            blocked_batches=blocked,
            ok_batches=ok,
            pending_count=pending,
            batches=statuses,
            has_blocked=blocked > 0,
            overall_status=overall
        )

    def analyze_orientation(self, task_id: str, force_flush: bool = True) -> TaskOrientationSummary:
        if force_flush:
            self.batch_manager.flush_task(task_id, force=True)
        return self.get_task_orientation_summary(task_id)
