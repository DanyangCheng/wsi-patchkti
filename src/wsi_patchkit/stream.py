"""Lazy conversion of patch requests into image arrays."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Literal

from .geometry import read_aligned_patch
from .io.base import SlideReader
from .types import Patch, PatchRequest, SlideMetadata


class PatchStream(Iterable[Patch]):
    """Read an iterable of requests while caching metadata per slide/MPP pair."""

    def __init__(
        self,
        reader: SlideReader,
        requests: Iterable[PatchRequest],
        *,
        interpolation: Literal["nearest", "bilinear"] = "bilinear",
        color_mode: Literal["rgb", "native"] = "rgb",
    ) -> None:
        self.reader = reader
        self.requests = requests
        self.interpolation = interpolation
        self.color_mode = color_mode

    def __iter__(self) -> Iterator[Patch]:
        metadata_cache: dict[tuple[str, tuple[float, float] | None], SlideMetadata] = {}
        for request in self.requests:
            key = request.slide, request.source_mpp
            metadata = metadata_cache.get(key)
            if metadata is None:
                metadata = self.reader.metadata(
                    request.slide,
                    source_mpp=request.source_mpp,
                )
                metadata_cache[key] = metadata
            yield Patch(
                request,
                read_aligned_patch(
                    self.reader,
                    request,
                    metadata=metadata,
                    interpolation=self.interpolation,
                    color_mode=self.color_mode,
                ),
            )
