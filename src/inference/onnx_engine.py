import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
from src.utils.logging import get_logger

logger = get_logger("inference.onnx_engine")

class ONNXInferenceEngine:
    def __init__(self,
                 model_path: str,
                 use_gpu: bool = True,
                 intra_op_num_threads: int = 8,
                 inter_op_num_threads: int = 4,
                 enable_cuda_graph: bool = True,
                 max_batch_size: int = 16):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.enable_cuda_graph = enable_cuda_graph
        self.max_batch_size = max_batch_size
        self._session = None
        self._input_name = None
        self._output_name = None
        self._input_shape = None
        self._warmup_done = False
        self._initialize_session()

    def _initialize_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        providers = []
        provider_options = []
        if self.use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            cuda_options = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            if self.enable_cuda_graph:
                cuda_options["enable_cuda_graph"] = True
            providers.append("CUDAExecutionProvider")
            provider_options.append(cuda_options)
            logger.info("Using CUDAExecutionProvider for ONNX inference")
        if 'CPUExecutionProvider' not in providers:
            providers.append("CPUExecutionProvider")
            cpu_options = {
                "intra_op_num_threads": self.intra_op_num_threads,
                "inter_op_num_threads": self.inter_op_num_threads,
                "arena_extend_strategy": "kSameAsRequested",
            }
            provider_options.append(cpu_options)
            logger.info(f"Using CPUExecutionProvider with {self.intra_op_num_threads} threads")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.intra_op_num_threads
        sess_options.inter_op_num_threads = self.inter_op_num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.enable_profiling = False
        sess_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options if len(provider_options) > 1 else None
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        logger.info(f"ONNX session initialized: input={self._input_name}, "
                    f"output={self._output_name}, shape={self._input_shape}")

    def _warmup(self, input_shape: Tuple[int, ...]) -> None:
        if self._warmup_done:
            return
        logger.info(f"Warming up ONNX engine with shape {input_shape}...")
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        for _ in range(3):
            _ = self._session.run([self._output_name], {self._input_name: dummy_input})
        self._warmup_done = True
        logger.info("ONNX engine warmup complete")

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        if input_array.ndim == 2:
            input_array = input_array[np.newaxis, np.newaxis, ...]
        elif input_array.ndim == 3:
            input_array = input_array[:, np.newaxis, ...]
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)
        batch_size = input_array.shape[0]
        if batch_size > self.max_batch_size:
            return self._infer_batched(input_array)
        self._warmup(input_array.shape)
        start_time = time.perf_counter()
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: input_array}
        )
        inference_time = time.perf_counter() - start_time
        logger.debug(f"ONNX inference: batch={batch_size}, time={inference_time*1000:.2f}ms, "
                    f"throughput={batch_size/inference_time:.2f} img/s")
        return outputs[0]

    def _infer_batched(self, input_array: np.ndarray) -> np.ndarray:
        batch_size = input_array.shape[0]
        outputs = []
        for start in range(0, batch_size, self.max_batch_size):
            end = min(start + self.max_batch_size, batch_size)
            batch = input_array[start:end]
            output = self._session.run(
                [self._output_name],
                {self._input_name: batch}
            )[0]
            outputs.append(output)
        return np.concatenate(outputs, axis=0)

    def infer_patches(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        if not patches:
            return []
        batch = np.stack(patches, axis=0)
        if batch.ndim == 3:
            batch = batch[:, np.newaxis, ...]
        outputs = self.infer(batch)
        return [outputs[i] for i in range(outputs.shape[0])]

    def benchmark(self, input_shape: Tuple[int, ...] = (1, 1, 512, 512),
                  num_runs: int = 100) -> Dict[str, float]:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        for _ in range(5):
            _ = self._session.run([self._output_name], {self._input_name: dummy_input})
        timings = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self._session.run([self._output_name], {self._input_name: dummy_input})
            timings.append(time.perf_counter() - start)
        timings_ms = np.array(timings) * 1000
        return {
            "mean_ms": float(np.mean(timings_ms)),
            "median_ms": float(np.median(timings_ms)),
            "min_ms": float(np.min(timings_ms)),
            "max_ms": float(np.max(timings_ms)),
            "std_ms": float(np.std(timings_ms)),
            "throughput_fps": float(input_shape[0] / np.mean(timings)),
        }

    def get_input_shape(self) -> List[int]:
        return self._input_shape

    def get_providers(self) -> List[str]:
        return self._session.get_providers() if self._session else []

    def close(self) -> None:
        if self._session is not None:
            self._session = None
            logger.info("ONNX session closed")

    def __del__(self) -> None:
        try:
            self.close()
        except:
            pass
