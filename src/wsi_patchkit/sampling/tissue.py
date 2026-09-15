"""Low-resolution tissue-mask mapping and request filtering."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray

from ..types import PatchRequest, as_size


@dataclass(slots=True)
class TissueMask:
    """A low-resolution binary mask mapped over a virtual WSI canvas."""

    mask: NDArray[np.generic]
    canvas_size: tuple[int, int]
    _integral: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError("mask must be a non-empty 2D array")
        self.mask = np.ascontiguousarray(mask)
        self.canvas_size = as_size(self.canvas_size, name="canvas_size")
        self._integral = np.pad(
            self.mask.astype(np.int64).cumsum(0).cumsum(1),
            ((1, 0), (1, 0)),
        )

    @classmethod
    def from_tiff(
        cls,
        path: str | Path,
        *,
        canvas_size: tuple[int, int],
    ) -> TissueMask:
        """Load a one-channel MINISBLACK or MINISWHITE TIFF mask."""
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            array = np.asarray(page.asarray())
            while array.ndim > 2 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.ndim != 2:
                raise ValueError("tissue mask TIFF must be single-channel")
            foreground = array == 0 if int(page.photometric) == 0 else array > 0
        return cls(foreground, canvas_size)

    def fraction(self, request: PatchRequest) -> float:
        """Approximate mask coverage within a request rectangle."""
        canvas_width, canvas_height = self.canvas_size
        x0, y0 = max(0, request.x), max(0, request.y)
        x1 = min(canvas_width, request.x + request.width)
        y1 = min(canvas_height, request.y + request.height)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        mask_height, mask_width = self.mask.shape
        mx0 = min(mask_width, x0 * mask_width // canvas_width)
        my0 = min(mask_height, y0 * mask_height // canvas_height)
        mx1 = min(mask_width, -(-x1 * mask_width // canvas_width))
        my1 = min(mask_height, -(-y1 * mask_height // canvas_height))
        if mx1 <= mx0 or my1 <= my0:
            return 0.0
        integral = self._integral
        count = (
            integral[my1, mx1]
            - integral[my0, mx1]
            - integral[my1, mx0]
            + integral[my0, mx0]
        )
        return float(count / ((mx1 - mx0) * (my1 - my0)))


class TissueFilter:
    """Filter requests using one tissue mask per slide."""

    def __init__(
        self,
        masks: Mapping[str | Path, TissueMask],
        *,
        minimum_fraction: float = 0.01,
        missing: str = "error",
    ) -> None:
        if not 0 <= minimum_fraction <= 1:
            raise ValueError("minimum_fraction must be in [0, 1]")
        if missing not in ("error", "keep", "drop"):
            raise ValueError("missing must be 'error', 'keep', or 'drop'")
        self.masks = {str(Path(path)): mask for path, mask in masks.items()}
        self.minimum_fraction = float(minimum_fraction)
        self.missing = missing

    def filter(self, requests: Iterable[PatchRequest]) -> Iterator[PatchRequest]:
        for request in requests:
            mask = self.masks.get(request.slide)
            if mask is None:
                if self.missing == "error":
                    raise KeyError(f"no tissue mask for slide {request.slide!r}")
                if self.missing == "keep":
                    yield request
                continue
            if mask.fraction(request) >= self.minimum_fraction:
                yield request
