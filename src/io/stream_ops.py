import os
import mmap
import numpy as np
from typing import Iterator, Tuple, Optional, Callable
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger("io.stream_ops")

def zero_copy_read(file_path: str,
                   offset: int = 0,
                   size: Optional[int] = None,
                   dtype: np.dtype = np.float32) -> np.ndarray:
    file_size = os.path.getsize(file_path)
    if size is None:
        size = file_size - offset
    if offset + size > file_size:
        raise ValueError(f"Read beyond file size: offset={offset}, size={size}, file_size={file_size}")
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ, offset=0) as mm:
            raw_data = mm[offset:offset + size]
            return np.frombuffer(raw_data, dtype=dtype).copy()

def stream_chunks(file_path: str,
                  chunk_size: int = 32 * 1024 * 1024,
                  offset: int = 0,
                  max_chunks: Optional[int] = None) -> Iterator[Tuple[int, bytes]]:
    file_size = os.path.getsize(file_path)
    bytes_read = offset
    chunks_yielded = 0
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ, offset=0) as mm:
            while bytes_read < file_size:
                if max_chunks is not None and chunks_yielded >= max_chunks:
                    break
                chunk_end = min(bytes_read + chunk_size, file_size)
                chunk_data = mm[bytes_read:chunk_end]
                yield bytes_read, chunk_data
                bytes_read = chunk_end
                chunks_yielded += 1

def stream_array_chunks(file_path: str,
                        dtype: np.dtype,
                        shape: Tuple[int, ...],
                        chunk_size: int = 32 * 1024 * 1024,
                        offset: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
    row_size = np.prod(shape[1:]) * np.dtype(dtype).itemsize
    rows_per_chunk = max(1, chunk_size // row_size)
    total_rows = shape[0]
    for start_row in range(0, total_rows, rows_per_chunk):
        end_row = min(start_row + rows_per_chunk, total_rows)
        num_rows = end_row - start_row
        start_byte = offset + start_row * row_size
        size = num_rows * row_size
        data = zero_copy_read(file_path, start_byte, size, dtype)
        yield start_row, data.reshape((num_rows,) + shape[1:])

def async_stream_process(file_path: str,
                         processor: Callable[[np.ndarray, int], None],
                         dtype: np.dtype,
                         shape: Tuple[int, ...],
                         chunk_size: int = 32 * 1024 * 1024,
                         offset: int = 0) -> None:
    for start_idx, chunk in stream_array_chunks(file_path, dtype, shape, chunk_size, offset):
        processor(chunk, start_idx)

def create_mrc_header_bytes(nx: int, ny: int, nz: int,
                            mode: int = 2,
                            pixel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                            dmin: float = 0.0, dmax: float = 1.0, dmean: float = 0.5) -> bytes:
    import struct
    header_format = "<" + "".join([
        "i" * 3, "i", "i" * 3, "i" * 3, "f" * 3, "f" * 3,
        "i" * 3, "f" * 3, "i" * 2, "100s", "f" * 3, "4s", "4s", "f", "i", "800s"
    ])
    header_data = [
        nx, ny, nz, mode, 0, 0, 0, nx, ny, nz,
        pixel_size[0] * nx, pixel_size[1] * ny, pixel_size[2] * nz,
        90.0, 90.0, 90.0, 1, 2, 3, dmin, dmax, dmean,
        0, 0, b'\x00' * 100, 0.0, 0.0, 0.0, b'MAP ',
        b'\x00\x00\x00\x00', 0.0, 0, b'\x00' * 800
    ]
    return struct.pack(header_format, *header_data)

def write_mrc_file(file_path: str, data: np.ndarray,
                   pixel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> str:
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    nz, ny, nx = data.shape
    mode_map = {np.int8: 0, np.int16: 1, np.float32: 2, np.float16: 4, np.uint16: 6}
    mode = mode_map.get(data.dtype.type, 2)
    if mode == 2 and data.dtype != np.float32:
        data = data.astype(np.float32)
    header = create_mrc_header_bytes(
        nx, ny, nz, mode, pixel_size,
        float(data.min()), float(data.max()), float(data.mean())
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())
    logger.info(f"Wrote MRC file: {file_path} | shape: {nx}x{ny}x{nz} | dtype: {data.dtype}")
    return file_path

write_mrc = write_mrc_file
