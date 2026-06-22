import os
import argparse
import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query, HTTPException, Depends, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List
from pathlib import Path
from src.utils.logging import get_logger
from src.utils.config import load_config
from src.api.schemas import (
    TaskStatus, PickingResult, TaskCreateResponse,
    TaskStatusResponse, SystemInfo, HealthResponse,
    PickingRequest, BenchmarkResult, ExportRequest
)
from src.api.service import PickingService

logger = get_logger("api.main")
config = load_config()
picking_service = PickingService()

def create_app() -> FastAPI:
    app = FastAPI(
        title=config["system"]["name"],
        version=config["system"]["version"],
        description="Cryo-EM Single Particle Intelligent Picking and Projection Unmixing Analysis System API",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["api"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    static_dir = Path("static")
    templates_dir = Path("templates")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    if templates_dir.exists():
        app.state.templates = Jinja2Templates(directory=str(templates_dir))
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting up Cryo-EM Particle Picker API...")
        onnx_path = config["inference"]["onnx_model_path"]
        if os.path.exists(onnx_path):
            picking_service.load_model(onnx_path)
        else:
            logger.warning(f"ONNX model not found at {onnx_path}. Model will be loaded on first request.")
        logger.info("API startup complete")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Cryo-EM Particle Picker API...")
        picking_service.cleanup()
        logger.info("API shutdown complete")

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def root():
        if hasattr(app.state, "templates"):
            return app.state.templates.TemplateResponse("index.html", {"request": {}})
        return HTMLResponse("""
        <html>
            <head><title>Cryo-EM Particle Picker</title></head>
            <body>
                <h1>Cryo-EM Single Particle Intelligent Picking System</h1>
                <p>API Documentation: <a href="/api/docs">/api/docs</a></p>
            </body>
        </html>
        """)

    @app.get("/api/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        return HealthResponse()

    @app.get("/api/system/info", response_model=SystemInfo, tags=["System"])
    async def get_system_info():
        return picking_service.get_system_info()

    @app.post("/api/system/benchmark", response_model=BenchmarkResult, tags=["System"])
    async def run_benchmark(
        width: int = Query(4096, gt=0, description="Image width"),
        height: int = Query(4096, gt=0, description="Image height"),
        num_runs: int = Query(10, ge=1, le=100, description="Number of benchmark runs")
    ):
        try:
            result = picking_service.run_benchmark((height, width), num_runs)
            return BenchmarkResult(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/pick/upload", response_model=TaskCreateResponse, tags=["Picking"])
    async def upload_and_pick(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
        min_distance: Optional[int] = Query(None, ge=1),
        max_particles: Optional[int] = Query(None, ge=1),
        export_format: Optional[str] = Query(None, pattern="^(star|csv|tsv|coords|npy)$")
    ):
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        allowed_extensions = ['.mrc', '.mrcs', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {allowed_extensions}"
            )
        file_path, file_size = await picking_service.save_uploaded_file(file)
        from src.io.mrc_parser import MRCStreamParser
        with MRCStreamParser(file_path, zero_copy=True) as parser:
            from src.api.schemas import ImageInfo
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
        task_id = picking_service.create_task(file.filename, image_info)
        request = PickingRequest(
            confidence_threshold=confidence_threshold,
            min_distance=min_distance,
            max_particles=max_particles,
            export_format=export_format
        )
        background_tasks.add_task(
            picking_service.process_task,
            task_id, file_path, request
        )
        return TaskCreateResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message=f"Task created. File size: {file_size/1024/1024:.2f}MB"
        )

    @app.get("/api/pick/tasks", response_model=List[PickingResult], tags=["Picking"])
    async def list_tasks(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        status: Optional[TaskStatus] = Query(None)
    ):
        return picking_service.list_tasks(skip, limit, status)

    @app.get("/api/pick/tasks/{task_id}", response_model=PickingResult, tags=["Picking"])
    async def get_task_result(task_id: str):
        task = picking_service.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task

    @app.get("/api/pick/tasks/{task_id}/status", response_model=TaskStatusResponse, tags=["Picking"])
    async def get_task_status(task_id: str):
        return picking_service.get_task_status(task_id)

    @app.get("/api/results/{task_id}/{file_type}", tags=["Results"])
    async def get_result_file(
        task_id: str,
        file_type: str = FastAPIPath(..., pattern="^(image|probability_map|coordinates)$")
    ):
        file_path = picking_service.get_result_file_path(task_id, file_type)
        filename = f"{task_id}_{file_type}" + os.path.splitext(file_path)[1]
        return FileResponse(
            file_path,
            media_type='application/octet-stream',
            filename=filename
        )

    @app.post("/api/results/{task_id}/export", tags=["Results"])
    async def export_coordinates(task_id: str, request: ExportRequest):
        try:
            file_path = picking_service.export_coordinates(task_id, request.format)
            filename = f"{task_id}_coordinates.{request.format}"
            return FileResponse(
                file_path,
                media_type='application/octet-stream',
                filename=filename
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/system/reload_model", tags=["System"])
    async def reload_model(model_path: Optional[str] = Query(None)):
        success = picking_service.load_model(model_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model")
        return {"status": "success", "message": "Model reloaded successfully"}

    return app

app = create_app()

def main():
    parser = argparse.ArgumentParser(description="Start Cryo-EM Particle Picker API Server")
    parser.add_argument("--host", type=str, default=config["api"]["host"],
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=config["api"]["port"],
                       help="Port to bind to")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    args = parser.parse_args()
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()
