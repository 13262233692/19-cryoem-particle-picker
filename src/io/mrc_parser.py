import os
import mmap
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Iterator
from enum import IntEnum
from src.utils.logging import get_logger

logger = get_logger("io.mrc_parser")

class MRCMode(IntEnum):
    INT8 = 0
    INT16 = 2
    FLOAT32 = 2
    COMPLEX16 = 3
    FLOAT16 = 4
    UINT16 = 6
    FLOAT64 = 7

MODE_DTYPE_MAP = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    3: np.complex64,
    4: np.float16,
    6: np.uint16,
    7: np.float64,
}

MODE_BYTES_PER_PIXEL = {
    0: 1,
    1: 2,
    2: 4,
    3: 8,
    4: 2,
    6: 2,
    7: 8,
}

HEADER_FIELDS = [
    ("nx", "i"), ("ny", "i"), ("nz", "i"),
    ("mode", "i"),
    ("nxstart", "i"), ("nystart", "i"), ("nzstart", "i"),
    ("mx", "i"), ("my", "i"), ("mz", "i"),
    ("cella_x", "f"), ("cella_y", "f"), ("cella_z", "f"),
    ("cellb_alpha", "f"), ("cellb_beta", "f"), ("cellb_gamma", "f"),
    ("mapc", "i"), ("mapr", "i"), ("maps", "i"),
    ("dmin", "f"), ("dmax", "f"), ("dmean", "f"),
    ("ispg", "i"), ("nsymbt", "i"),
    ("extra_1", "100s"),
    ("xorigin", "f"), ("yorigin", "f"), ("zorigin", "f"),
    ("map", "4s"),
    ("machst", "4s"),
    ("rms", "f"),
    ("nlabels", "i"),
    ("labels", "800s"),
]

HEADER_FORMAT = "<" + "".join(fmt for _, fmt in HEADER_FIELDS)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 1024, f"MRC header size should be 1024, got {HEADER_SIZE}"

@dataclass
class MRCHeader:
    nx: int = 0
    ny: int = 0
    nz: int = 0
    mode: int = 2
    nxstart: int = 0
    nystart: int = 0
    nzstart: int = 0
    mx: int = 0
    my: int = 0
    mz: int = 0
    cella_x: float = 0.0
    cella_y: float = 0.0
    cella_z: float = 0.0
    cellb_alpha: float = 0.0
    cellb_beta: float = 0.0
    cellb_gamma: float = 0.0
    mapc: int = 1
    mapr: int = 2
    maps: int = 3
    dmin: float = 0.0
    dmax: float = 0.0
    dmean: float = 0.0
    ispg: int = 0
    nsymbt: int = 0
    xorigin: float = 0.0
    yorigin: float = 0.0
    zorigin: float = 0.0
    map: bytes = b"MAP "
    machst: bytes = b""
    rms: float = 0.0
    nlabels: int = 0
    labels: bytes = b""
    pixel_size: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 1.0, 1.0))

    @classmethod
    def from_bytes(cls, header_bytes: bytes) -> "MRCHeader":
        if len(header_bytes) < HEADER_SIZE:
            raise ValueError(f"Header too short: {len(header_bytes)} < {HEADER_SIZE}")
        unpacked = struct.unpack(HEADER_FORMAT, header_bytes[:HEADER_SIZE])
        field_values = {name: unpacked[i] for i, (name, _) in enumerate(HEADER_FIELDS)}
        header = cls(**{k: v for k, v in field_values.items() if k in cls.__dataclass_fields__})
        if header.mx > 0 and header.my > 0 and header.mz > 0:
            header.pixel_size = (
                header.cella_x / header.mx,
                header.cella_y / header.my,
                header.cella_z / header.mz,
            )
        return header

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": (self.nx, self.ny, self.nz),
            "mode": self.mode,
            "pixel_size": self.pixel_size,
            "data_range": (self.dmin, self.dmax),
            "data_mean": self.dmean,
            "origin": (self.xorigin, self.yorigin, self.zorigin),
        }

class MRCStreamParser:
    def __init__(self, file_path: str, zero_copy: bool = True, chunk_size_mb: int = 32):
        self.file_path = file_path
        self.zero_copy = zero_copy
        self.chunk_size = chunk_size_mb * 1024 * 1024
        self._mmap = None
        self._file = None
        self.header: Optional[MRCHeader] = None
        self._data_offset = HEADER_SIZE
        self._dtype = None
        self._bytes_per_pixel = 0
        self._image_size = 0
        self._total_images = 0
        self._open()

    def _open(self) -> None:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"MRC file not found: {self.file_path}")
        file_size = os.path.getsize(self.file_path)
        if file_size < HEADER_SIZE:
            raise ValueError(f"File too small to be valid MRC: {file_size} bytes")
        self._file = open(self.file_path, "rb")
        header_bytes = self._file.read(HEADER_SIZE)
        self.header = MRCHeader.from_bytes(header_bytes)
        if self.header.mode not in MODE_DTYPE_MAP:
            raise ValueError(f"Unsupported MRC mode: {self.header.mode}")
        self._dtype = MODE_DTYPE_MAP[self.header.mode]
        self._bytes_per_pixel = MODE_BYTES_PER_PIXEL[self.header.mode]
        self._data_offset = HEADER_SIZE + self.header.nsymbt
        self._image_size = self.header.nx * self.header.ny * self._bytes_per_pixel
        expected_data_size = self.header.nx * self.header.ny * self.header.nz * self._bytes_per_pixel
        self._total_images = self.header.nz
        if file_size < self._data_offset + expected_data_size:
            logger.warning(f"File size smaller than expected. Truncated MRC?")
        if self.zero_copy:
            self._mmap = mmap.mmap(
                self._file.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
                offset=0
            )
        logger.info(f"Opened MRC: {self.file_path} | "
                    f"Dimensions: {self.header.nx}x{self.header.ny}x{self.header.nz} | "
                    f"Mode: {self.header.mode} ({self._dtype.__name__}) | "
                    f"Zero-copy: {self.zero_copy}")

    def get_image(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._total_images:
            raise IndexError(f"Image index {index} out of range [0, {self._total_images})")
        start_byte = self._data_offset + index * self._image_size
        end_byte = start_byte + self._image_size
        if self.zero_copy and self._mmap is not None:
            raw_data = self._mmap[start_byte:end_byte]
        else:
            self._file.seek(start_byte)
            raw_data = self._file.read(self._image_size)
        array = np.frombuffer(raw_data, dtype=self._dtype)
        return array.reshape((self.header.ny, self.header.nx)).copy()

    def get_image_batch(self, start_idx: int, end_idx: int) -> np.ndarray:
        if start_idx < 0 or end_idx > self._total_images or start_idx >= end_idx:
            raise IndexError(f"Invalid batch range: [{start_idx}, {end_idx})")
        num_images = end_idx - start_idx
        start_byte = self._data_offset + start_idx * self._image_size
        total_bytes = num_images * self._image_size
        if self.zero_copy and self._mmap is not None:
            raw_data = self._mmap[start_byte:start_byte + total_bytes]
        else:
            self._file.seek(start_byte)
            raw_data = self._file.read(total_bytes)
        array = np.frombuffer(raw_data, dtype=self._dtype)
        return array.reshape((num_images, self.header.ny, self.header.nx)).copy()

    def iterate_images(self, batch_size: int = 1) -> Iterator[np.ndarray]:
        for start in range(0, self._total_images, batch_size):
            end = min(start + batch_size, self._total_images)
            yield self.get_image_batch(start, end)

    def get_region(self, image_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        if image_idx < 0 or image_idx >= self._total_images:
            raise IndexError(f"Image index out of range")
        if x < 0 or y < 0 or x + w > self.header.nx or y + h > self.header.ny:
            raise ValueError(f"Region out of bounds: ({x},{y})+({w},{h})")
        img_start = self._data_offset + image_idx * self._image_size
        row_bytes = self.header.nx * self._bytes_per_pixel
        region_bytes = w * self._bytes_per_pixel
        result = np.empty((h, w), dtype=self._dtype)
        for row in range(h):
            start_byte = img_start + (y + row) * row_bytes + x * self._bytes_per_pixel
            end_byte = start_byte + region_bytes
            if self.zero_copy and self._mmap is not None:
                row_data = self._mmap[start_byte:end_byte]
            else:
                self._file.seek(start_byte)
                row_data = self._file.read(region_bytes)
            result[row] = np.frombuffer(row_data, dtype=self._dtype)
        return result

    def stream_chunks(self) -> Iterator[Tuple[int, np.ndarray]]:
        chunk_images = max(1, self.chunk_size // self._image_size)
        for start in range(0, self._total_images, chunk_images):
            end = min(start + chunk_images, self._total_images)
            yield start, self.get_image_batch(start, end)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.header.nz, self.header.ny, self.header.nx)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def num_images(self) -> int:
        return self._total_images

    def __len__(self) -> int:
        return self._total_images

    def __getitem__(self, idx: int) -> np.ndarray:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._total_images)
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            return self.get_image_batch(start, stop)
        return self.get_image(idx)

    def __enter__(self) -> "MRCStreamParser":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        logger.info(f"Closed MRC file: {self.file_path}")

    def __del__(self) -> None:
        try:
            self.close()
        except:
            pass
