"""Automatic routing between bundled WSI reader backends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from numpy.typing import NDArray

from ..types import SlideMetadata
from .base import SlideReader
from .openslide import OpenSlideReader
from .tiff import TiffReader

_TIFF_SUFFIXES = {".tif", ".tiff", ".btf", ".btiff"}


class AutoSlideReader:
    """Route generic TIFF files to tifffile and other WSIs to OpenSlide.

    OpenSlide is imported lazily, so TIFF-only applications do not need the
    optional native dependency.
    """

    def __init__(self, cache_size: int = 4) -> None:
        self.cache_size = int(cache_size)
        self._tiff = TiffReader(cache_size=cache_size)
        self._openslide: OpenSlideReader | None = None

    def _reader(self, path: str | Path) -> SlideReader:
        if Path(path).suffix.lower() in _TIFF_SUFFIXES:
            return self._tiff
        if self._openslide is None:
            self._openslide = OpenSlideReader(cache_size=self.cache_size)
        return self._openslide

    def metadata(
        self,
        path: str | Path,
        *,
        source_mpp: float | Sequence[float] | None = None,
    ) -> SlideMetadata:
        return self._reader(path).metadata(path, source_mpp=source_mpp)

    def read_region(
        self,
        path: str | Path,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> NDArray:
        return self._reader(path).read_region(path, location, level, size)

    def close(self) -> None:
        self._tiff.close()
        if self._openslide is not None:
            self._openslide.close()
            self._openslide = None

    def __enter__(self) -> AutoSlideReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

