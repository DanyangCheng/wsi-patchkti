"""Sampling interfaces and shared helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ..types import PatchRequest, SamplingContext, SlideSpec


class PatchSampler(Protocol):
    """Protocol implemented by slide-aware request samplers."""

    def sample(
        self,
        slides: Iterable[SlideSpec],
        *,
        context: SamplingContext | None = None,
    ) -> Iterable[PatchRequest]:
        """Return requests assigned to the supplied worker context."""
