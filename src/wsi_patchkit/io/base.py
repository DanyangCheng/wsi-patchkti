"""Common reader protocol."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..types import SlideMetadata


@runtime_checkable
class SlideReader(Protocol):
    """OpenSlide-compatible region reader.

    ``location`` is expressed in level-0 pixels. ``size`` is expressed in pixels
    of ``level``. Calls are expected to be in bounds; aligned patch helpers own
    clipping and padding.
    """

    def metadata(
        self,
        path: str | Path,
        *,
        source_mpp: float | Sequence[float] | None = None,
    ) -> SlideMetadata:
        """Return pyramid and physical-resolution metadata."""

    def read_region(
        self,
        path: str | Path,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> NDArray[np.generic]:
        """Read an in-bounds region as an HWC array."""

    def close(self) -> None:
        """Release open slide handles."""
