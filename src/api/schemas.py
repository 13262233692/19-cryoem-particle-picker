from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import enum

class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Particle(BaseModel):
    x: float = Field(..., description="Particle X coordinate")
    y: float = Field(..., description="Particle Y coordinate")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    radius: Optional[float] = Field(None, description="Particle radius")
    snr: Optional[float] = Field(None, description="Signal-to-noise ratio")
    eccentricity: Optional[float] = Field(None, description="Eccentricity")
    orientation: Optional[float] = Field(None, description="Orientation angle")

class ImageInfo(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    pixel_size: Tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0))
    data_type: str = Field(default="float32")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None

class ProcessingTimes(BaseModel):
    preprocessing: float = Field(..., ge=0.0, description="Preprocessing time in seconds")
    inference: float = Field(..., ge=0.0, description="Inference time in seconds")
    postprocessing: float = Field(..., ge=0.0, description="Postprocessing time in seconds")
    total: float = Field(..., ge=0.0, description="Total processing time in seconds")

class PickingResult(BaseModel):
    task_id: str = Field(...)
    status: TaskStatus = Field(...)
    file_name: str = Field(...)
    image_info: ImageInfo = Field(...)
    num_particles: int = Field(..., ge=0)
    particles: List[Particle] = Field(default_factory=list)
    processing_times: Optional[ProcessingTimes] = None
    result_image_url: Optional[str] = None
    probability_map_url: Optional[str] = None
    coordinates_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @field_validator('particles')
    @classmethod
    def sort_particles_by_score(cls, v):
        return sorted(v, key=lambda p: p.score, reverse=True)

class TaskCreateResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class SystemInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    version: str
    device: str
    model_loaded: bool
    model_path: Optional[str] = None
    gpu_available: bool
    gpu_memory: Optional[float] = None
    total_memory: Optional[float] = None
    cpu_count: int
    uptime: float

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BenchmarkResult(BaseModel):
    image_shape: Tuple[int, int]
    mean_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_fps: float
    avg_particles: float

class ExportRequest(BaseModel):
    task_id: str
    format: str = Field(default="star", pattern="^(star|csv|tsv|coords|npy)$")
    include_quality: bool = Field(default=True)

class PickingRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_distance: Optional[int] = Field(None, ge=1)
    max_particles: Optional[int] = Field(None, ge=1)
    use_preprocessing: bool = Field(default=True)
    export_format: Optional[str] = Field(None, pattern="^(star|csv|tsv|coords|npy)$")


class EulerHistogram(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    phi_bins: List[float] = Field(default_factory=list)
    phi_counts: List[int] = Field(default_factory=list)
    theta_bins: List[float] = Field(default_factory=list)
    theta_counts: List[int] = Field(default_factory=list)
    psi_bins: List[float] = Field(default_factory=list)
    psi_counts: List[int] = Field(default_factory=list)
    n_orientations: int = 0


class OrientationAnalysis(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    particle_count: int
    mean_residual: float
    median_residual: float
    max_residual: float
    min_residual: float
    angular_spread_deg: float
    orientation_coverage: float
    is_preferred_orientation: bool
    euler_histogram: EulerHistogram
    processing_time_ms: float
    common_line_residuals: Optional[List[float]] = None


class PreferredOrientationAlert(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    alert_id: str
    task_id: str
    batch_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_preferred_orientation: bool
    block_reason: str
    particle_count: int
    orientation: OrientationAnalysis
    severity: str = Field(default="critical")
    message: str = Field(
        default="优势取向报废警示：该批次蛋白颗粒在冰层中呈现绝对单一优势取向，"
                "三维重构空间完备性丧失。样品已被静默拦截，后续高分辨率迭代已终止。"
    )
    action_required: bool = Field(default=True)


class OrientationAnalysisRequest(BaseModel):
    task_id: str
    force_flush: bool = Field(default=True)
    particle_ids: Optional[List[int]] = None


class BatchStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    batch_id: str
    task_id: str
    particle_count: int
    is_blocked: bool
    block_reason: Optional[str] = None
    analyzed: bool
    is_preferred_orientation: Optional[bool] = None
    orientation: Optional[OrientationAnalysis] = None
    created_at: datetime
    analyzed_at: Optional[datetime] = None


class TaskOrientationSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    task_id: str
    total_particles_captured: int
    total_batches: int
    blocked_batches: int
    ok_batches: int
    pending_count: int
    batches: List[BatchStatus] = Field(default_factory=list)
    has_blocked: bool = Field(default=False)
    overall_status: str = Field(default="ok")

