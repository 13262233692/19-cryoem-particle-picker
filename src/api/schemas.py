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
