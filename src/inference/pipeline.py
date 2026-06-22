import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
from src.utils.logging import get_logger
from src.utils.config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from src.inference.onnx_engine import ONNXInferenceEngine
from src.postprocessing.detection import PeakDetector, NMS, extract_coordinates

logger = get_logger("inference.pipeline")

@dataclass
class InferenceResult:
    coordinates: List[Tuple[int, int]]
    confidence_scores: List[float]
    segmentation_mask: np.ndarray
    probability_map: np.ndarray
    preprocessed_image: np.ndarray
    original_image: np.ndarray
    processing_time: Dict[str, float]
    num_particles: int

class InferencePipeline:
    def __init__(self,
                 config_path: str = "configs/config.yaml",
                 onnx_model_path: Optional[str] = None,
                 preprocessing_pipeline: Optional[PreprocessingPipeline] = None):
        self.config = load_config(config_path)
        if onnx_model_path is None:
            onnx_model_path = self.config["inference"]["onnx_model_path"]
        self.onnx_engine = ONNXInferenceEngine(
            model_path=onnx_model_path,
            use_gpu=self.config["inference"]["use_gpu"],
            intra_op_num_threads=self.config["inference"]["intra_op_num_threads"],
            inter_op_num_threads=self.config["inference"]["inter_op_num_threads"],
            enable_cuda_graph=self.config["inference"]["enable_cuda_graph"],
        )
        if preprocessing_pipeline is None:
            preproc_config = self.config["preprocessing"]
            self.preprocessing = PreprocessingPipeline(
                clahe_clip_limit=preproc_config["clahe"]["clip_limit"],
                clahe_tile_grid=tuple(preproc_config["clahe"]["tile_grid_size"]),
                bandpass_low_sigma=preproc_config["bandpass_filter"]["low_sigma"],
                bandpass_high_sigma=preproc_config["bandpass_filter"]["high_sigma"],
                percentile_low=preproc_config["normalize"]["percentile_low"],
                percentile_high=preproc_config["normalize"]["percentile_high"],
                patch_size=preproc_config["patch_size"],
                overlap=preproc_config["overlap"],
                apply_clahe=True,
                apply_bandpass=True
            )
        else:
            self.preprocessing = preprocessing_pipeline
        self.peak_detector = PeakDetector(
            min_distance=self.config["postprocessing"]["min_distance"],
            threshold_abs=self.config["postprocessing"]["threshold_abs"],
            exclude_border=self.config["postprocessing"]["exclude_border"],
            max_particles=self.config["postprocessing"]["max_particles"]
        )
        self.nms = NMS(
            nms_threshold=self.config["inference"]["nms_threshold"],
            min_particle_size=self.config["inference"]["min_particle_size"],
            max_particle_size=self.config["inference"]["max_particle_size"]
        )
        self.confidence_threshold = self.config["inference"]["confidence_threshold"]
        self.patch_size = self.config["preprocessing"]["patch_size"]
        self.overlap = self.config["preprocessing"]["overlap"]
        logger.info("Inference pipeline initialized")

    def _extract_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        h, w = image.shape
        step = self.patch_size - self.overlap
        patches = []
        positions = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                y_start = max(0, y_end - self.patch_size)
                x_start = max(0, x_end - self.patch_size)
                patch = image[y_start:y_end, x_start:x_end]
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    padded = np.zeros((self.patch_size, self.patch_size), dtype=patch.dtype)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((x_start, y_start, x_end, y_end))
        return patches, positions

    def _merge_probability_maps(self,
                                patch_outputs: List[np.ndarray],
                                positions: List[Tuple[int, int, int, int]],
                                output_shape: Tuple[int, int]) -> np.ndarray:
        h, w = output_shape
        merged = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        half_overlap = self.overlap // 2
        for output, (x_start, y_start, x_end, y_end) in zip(patch_outputs, positions):
            if output.ndim == 4:
                output = output[0]
            if output.shape[0] > 1:
                prob = output[1]
            else:
                prob = 1 / (1 + np.exp(-output[0]))
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            prob_cropped = prob[:actual_h, :actual_w]
            weight = np.ones((actual_h, actual_w), dtype=np.float32)
            if half_overlap > 0:
                fade_y = np.ones(actual_h, dtype=np.float32)
                fade_x = np.ones(actual_w, dtype=np.float32)
                if y_start > 0:
                    fade_y[:half_overlap] = np.linspace(0, 1, half_overlap)
                if y_end < h:
                    fade_y[-half_overlap:] = np.linspace(1, 0, half_overlap)
                if x_start > 0:
                    fade_x[:half_overlap] = np.linspace(0, 1, half_overlap)
                if x_end < w:
                    fade_x[-half_overlap:] = np.linspace(1, 0, half_overlap)
                weight = fade_y[:, np.newaxis] * fade_x[np.newaxis, :]
            merged[y_start:y_end, x_start:x_end] += prob_cropped * weight
            weights[y_start:y_end, x_start:x_end] += weight
        weights[weights == 0] = 1
        return merged / weights

    def _process_single_patch(self, image: np.ndarray) -> InferenceResult:
        times = {}
        start_total = time.perf_counter()
        original = image.copy()
        t0 = time.perf_counter()
        preproc_result = self.preprocessing.process(image, use_patching=False)
        preprocessed = preproc_result.image
        times["preprocessing"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        input_tensor = preprocessed[np.newaxis, np.newaxis, ...].astype(np.float32)
        output = self.onnx_engine.infer(input_tensor)
        times["inference"] = time.perf_counter() - t1
        if output.shape[1] > 1:
            prob_map = np.exp(output[:, 1, :, :]) / np.sum(np.exp(output), axis=1)
            prob_map = prob_map[0]
        else:
            prob_map = 1 / (1 + np.exp(-output[0, 0]))
        seg_mask = prob_map > self.confidence_threshold
        t2 = time.perf_counter()
        coords, scores = self.peak_detector.detect(prob_map)
        coords, scores = self.nms(coords, scores, prob_map.shape)
        times["postprocessing"] = time.perf_counter() - t2
        times["total"] = time.perf_counter() - start_total
        return InferenceResult(
            coordinates=coords,
            confidence_scores=scores,
            segmentation_mask=seg_mask,
            probability_map=prob_map,
            preprocessed_image=preprocessed,
            original_image=original,
            processing_time=times,
            num_particles=len(coords)
        )

    def process(self, image: np.ndarray, use_patching: Optional[bool] = None) -> InferenceResult:
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            elif image.shape[-1] == 1:
                image = image.squeeze(-1)
        h, w = image.shape
        if use_patching is None:
            use_patching = h > self.patch_size or w > self.patch_size
        if not use_patching:
            return self._process_single_patch(image)
        times = {}
        start_total = time.perf_counter()
        original = image.copy()
        t0 = time.perf_counter()
        preproc_result = self.preprocessing.process(image, use_patching=False)
        preprocessed = preproc_result.image
        times["preprocessing"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        patches, positions = self._extract_patches(preprocessed)
        patch_outputs = self.onnx_engine.infer_patches(patches)
        prob_map = self._merge_probability_maps(patch_outputs, positions, (h, w))
        times["inference"] = time.perf_counter() - t1
        seg_mask = prob_map > self.confidence_threshold
        t2 = time.perf_counter()
        coords, scores = self.peak_detector.detect(prob_map)
        coords, scores = self.nms(coords, scores, prob_map.shape)
        times["postprocessing"] = time.perf_counter() - t2
        times["total"] = time.perf_counter() - start_total
        logger.info(f"Inference complete: {len(coords)} particles detected, "
                   f"total_time={times['total']*1000:.2f}ms")
        return InferenceResult(
            coordinates=coords,
            confidence_scores=scores,
            segmentation_mask=seg_mask,
            probability_map=prob_map,
            preprocessed_image=preprocessed,
            original_image=original,
            processing_time=times,
            num_particles=len(coords)
        )

    def process_batch(self, images: List[np.ndarray]) -> List[InferenceResult]:
        results = []
        for img in images:
            results.append(self.process(img))
        return results

    def benchmark(self, image_shape: Tuple[int, int] = (4096, 4096),
                  num_runs: int = 10) -> Dict[str, Any]:
        dummy_image = np.random.randn(*image_shape).astype(np.float32)
        timings = []
        particle_counts = []
        for _ in range(num_runs):
            result = self.process(dummy_image)
            timings.append(result.processing_time["total"])
            particle_counts.append(result.num_particles)
        timings_ms = np.array(timings) * 1000
        return {
            "image_shape": image_shape,
            "mean_time_ms": float(np.mean(timings_ms)),
            "median_time_ms": float(np.median(timings_ms)),
            "min_time_ms": float(np.min(timings_ms)),
            "max_time_ms": float(np.max(timings_ms)),
            "throughput_fps": float(1.0 / np.mean(timings)),
            "avg_particles": float(np.mean(particle_counts)),
            "onnx_benchmark": self.onnx_engine.benchmark()
        }

    def close(self) -> None:
        self.onnx_engine.close()
        logger.info("Inference pipeline closed")

    def __enter__(self) -> "InferencePipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
