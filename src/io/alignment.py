"""
Axis Alignment Transformer for Cryo-EM Micrographs

Implements scanning spectrum-order adaptive rearrangement operators
that dynamically sniff the mapc/mapr/maps axis-order metadata flags
in the MRC volume header and perform context-aware tensor reordering
before feeding into the downstream inference pipeline.

This eliminates the hard-coded horizontal-row-scan assumption which
causes catastrophic feature misalignment when electron-damaged
fibrillar protein datasets record fast-axis scans in the
inverted vertical direction (bottom-to-top, mapc=2, mapr=1).
"""

import numpy as np
import gc
import weakref
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from src.utils.logging import get_logger

logger = get_logger("io.alignment")

MRC_AXIS_X = 1
MRC_AXIS_Y = 2
MRC_AXIS_Z = 3

CANONICAL_ORDER = (MRC_AXIS_X, MRC_AXIS_Y, MRC_AXIS_Z)

@dataclass
class AxisReorderPlan:
    original_mapc: int
    original_mapr: int
    original_maps: int
    transpose_2d: Optional[Tuple[int, int]] = None
    flip_lr: bool = False
    flip_ud: bool = False
    rotate_k: int = 0
    is_canonical: bool = True
    description: str = "Canonical (X->columns, Y->rows, Z->sections)"
    _inverse_ops: List[Tuple[str, Any]] = field(default_factory=list)

    @property
    def requires_transform(self) -> bool:
        return not self.is_canonical


class AxisAlignmentTransformer:
    """
    Scanning spectrum-order adaptive rearrangement operator.

    Dynamically sniffs mapc/mapr/maps axis-order metadata from the
    MRC header and builds a reorder plan that brings non-standard
    scan arrangements into the canonical (fast-X, slow-Y, slowest-Z)
    ordering expected by the downstream residual kernels.

    Usage:
        transformer = AxisAlignmentTransformer()
        plan = transformer.build_plan(header.mapc, header.mapr, header.maps)
        canonical_image = transformer.reorder_image(raw_image, plan)
        # After particle picking...
        original_coords = transformer.invert_coordinates(picked_coords, plan, image_shape)
    """

    def __init__(self, enable_caching: bool = True, cache_max_entries: int = 64):
        self._cache: Dict[Tuple[int, int, int], AxisReorderPlan] = {}
        self._enable_caching = enable_caching
        self._cache_max_entries = cache_max_entries
        logger.info("AxisAlignmentTransformer initialized: "
                   f"caching={'on' if enable_caching else 'off'}, "
                   f"cache_limit={cache_max_entries}")

    def _evict_cache_if_needed(self) -> None:
        if len(self._cache) > self._cache_max_entries:
            self._cache.clear()
            gc.collect()

    def build_plan(self, mapc: int, mapr: int, maps: int = 3) -> AxisReorderPlan:
        key = (mapc, mapr, maps)
        if self._enable_caching and key in self._cache:
            return self._cache[key]

        self._evict_cache_if_needed()

        plan = self._compute_reorder_plan(mapc, mapr, maps)

        if self._enable_caching:
            self._cache[key] = plan

        return plan

    def _compute_reorder_plan(self, mapc: int, mapr: int, maps: int) -> AxisReorderPlan:
        plan = AxisReorderPlan(
            original_mapc=mapc,
            original_mapr=mapr,
            original_maps=maps
        )
        inverse_ops: List[Tuple[str, Any]] = []

        if (mapc, mapr, maps) == CANONICAL_ORDER:
            plan.is_canonical = True
            plan.description = "Canonical scan order (mapc=X, mapr=Y, maps=Z): " \
                             "Horizontal row-major, top-left origin"
            plan._inverse_ops = []
            return plan

        plan.is_canonical = False

        col_axis = mapc
        row_axis = mapr

        needs_transpose = (col_axis == MRC_AXIS_Y and row_axis == MRC_AXIS_X)

        desc_parts = []

        if needs_transpose:
            plan.transpose_2d = (1, 0)
            desc_parts.append("Transposed (columns<->rows)")
            inverse_ops.append(("transpose", (1, 0)))

            col_axis, row_axis = row_axis, col_axis

        sign_map = self._infer_axis_sign(mapc, mapr, maps)
        if sign_map.get(MRC_AXIS_X, 1) == -1:
            plan.flip_lr = True
            desc_parts.append("Horizontally flipped (X reversed)")
            inverse_ops.append(("flip_lr", True))

        if sign_map.get(MRC_AXIS_Y, 1) == -1:
            plan.flip_ud = True
            desc_parts.append("Vertically flipped (Y inverted, bottom-up scan)")
            inverse_ops.append(("flip_ud", True))

        plan._inverse_ops = list(reversed(inverse_ops))

        if desc_parts:
            plan.description = " | ".join(desc_parts)
        else:
            plan.description = f"Non-standard axis order: mapc={mapc}, mapr={mapr}, maps={maps}"

        logger.info(f"Axis reorder plan built: {plan.description}")
        return plan

    def _infer_axis_sign(self, mapc: int, mapr: int, maps: int) -> Dict[int, int]:
        sign_map = {MRC_AXIS_X: 1, MRC_AXIS_Y: 1, MRC_AXIS_Z: 1}
        if mapr > MRC_AXIS_Y:
            sign_map[MRC_AXIS_Y] = -1
        if mapc < 0:
            sign_map[MRC_AXIS_X] = -1
            sign_map[abs(mapc)] = -sign_map.get(abs(mapc), 1)
        if mapr < 0:
            sign_map[MRC_AXIS_Y] = -1
            sign_map[abs(mapr)] = -sign_map.get(abs(mapr), 1)
        return sign_map

    def reorder_image(self,
                      image: np.ndarray,
                      plan: AxisReorderPlan,
                      make_contiguous: bool = True) -> np.ndarray:
        if not plan.requires_transform:
            if make_contiguous and not image.flags.c_contiguous:
                return np.ascontiguousarray(image)
            return image

        result = image

        if plan.transpose_2d is not None:
            axes = plan.transpose_2d
            if result.ndim == 3:
                axes = (0,) + tuple(a + 1 for a in axes)
            result = np.transpose(result, axes)

        if plan.flip_lr:
            if result.ndim == 2:
                result = result[:, ::-1]
            elif result.ndim == 3:
                result = result[:, :, ::-1]

        if plan.flip_ud:
            if result.ndim == 2:
                result = result[::-1, :]
            elif result.ndim == 3:
                result = result[:, ::-1, :]

        if plan.rotate_k != 0:
            if result.ndim == 2:
                result = np.rot90(result, k=plan.rotate_k)
            elif result.ndim == 3:
                result = np.array([np.rot90(s, k=plan.rotate_k) for s in result])

        if make_contiguous and not result.flags.c_contiguous:
            result = np.ascontiguousarray(result)

        return result

    def invert_coordinates(self,
                           coordinates: List[Tuple[float, float]],
                           plan: AxisReorderPlan,
                           image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        if not plan.requires_transform or not coordinates:
            return list(coordinates)

        h, w = image_shape
        result = [(float(x), float(y)) for x, y in coordinates]

        if plan.rotate_k != 0:
            k = (-plan.rotate_k) % 4
            new_result = []
            for x, y in result:
                for _ in range(k):
                    x, y = y, (w - 1) - x
                new_result.append((x, y))
            result = new_result
            if k % 2 == 1:
                h, w = w, h

        if plan.flip_ud:
            result = [(x, (h - 1) - y) for x, y in result]

        if plan.flip_lr:
            result = [((w - 1) - x, y) for x, y in result]

        if plan.transpose_2d is not None:
            result = [(y, x) for x, y in result]

        return result

    def reorder_coordinates(self,
                            coordinates: List[Tuple[float, float]],
                            plan: AxisReorderPlan,
                            image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        if not plan.requires_transform or not coordinates:
            return list(coordinates)

        h, w = image_shape
        result = [(float(x), float(y)) for x, y in coordinates]

        if plan.transpose_2d is not None:
            result = [(y, x) for x, y in result]
            h, w = w, h

        if plan.flip_lr:
            result = [((w - 1) - x, y) for x, y in result]

        if plan.flip_ud:
            result = [(x, (h - 1) - y) for x, y in result]

        if plan.rotate_k != 0:
            new_result = []
            for x, y in result:
                for _ in range(plan.rotate_k % 4):
                    x, y = (h - 1) - y, x
                new_result.append((x, y))
            result = new_result

        return result

    def cleanup(self) -> None:
        self._cache.clear()
        gc.collect()

    def __len__(self) -> int:
        return len(self._cache)

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


_DEFAULT_TRANSFORMER: Optional[AxisAlignmentTransformer] = None


def get_default_transformer() -> AxisAlignmentTransformer:
    global _DEFAULT_TRANSFORMER
    if _DEFAULT_TRANSFORMER is None:
        _DEFAULT_TRANSFORMER = AxisAlignmentTransformer(
            enable_caching=True,
            cache_max_entries=128
        )
    return _DEFAULT_TRANSFORMER


def transform_mrc_image(image: np.ndarray,
                        mapc: int,
                        mapr: int,
                        maps: int = 3,
                        transformer: Optional[AxisAlignmentTransformer] = None
                        ) -> Tuple[np.ndarray, AxisReorderPlan]:
    if transformer is None:
        transformer = get_default_transformer()
    plan = transformer.build_plan(mapc, mapr, maps)
    transformed = transformer.reorder_image(image, plan)
    return transformed, plan


def is_vertical_inverted_scan(mapc: int, mapr: int, maps: int = 3) -> bool:
    abs_mapc = abs(mapc)
    abs_mapr = abs(mapr)
    if (abs_mapc == MRC_AXIS_Y and abs_mapr == MRC_AXIS_X):
        return True
    if abs_mapr > MRC_AXIS_Y:
        return True
    if mapr < 0 or mapc < 0:
        return True
    return False


def get_axis_order_summary(mapc: int, mapr: int, maps: int = 3) -> Dict[str, Any]:
    axis_names = {1: "X (horizontal)", 2: "Y (vertical)", 3: "Z (section)"}
    return {
        "columns_axis": axis_names.get(abs(mapc), f"Unknown({mapc})"),
        "rows_axis": axis_names.get(abs(mapr), f"Unknown({mapr})"),
        "sections_axis": axis_names.get(abs(maps), f"Unknown({maps})"),
        "columns_reversed": mapc < 0,
        "rows_reversed": mapr < 0,
        "sections_reversed": maps < 0,
        "is_vertical_scan": is_vertical_inverted_scan(mapc, mapr, maps),
        "is_canonical": (mapc, mapr, maps) == CANONICAL_ORDER
    }
