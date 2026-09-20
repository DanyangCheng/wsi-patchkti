"""Sampling interfaces and shared helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ..types import PatchRequest, SamplingContext, SlideSpec


class PatchSampler(Protocol):
    """Protocol implemented by slide-aware request samplers.

    Finite samplers should enumerate their global sequence through
    :meth:`SamplingContext.indices` and resolve padded positions with
    :meth:`SamplingContext.source_index`. This preserves deterministic,
    equal-length distributed sharding when requested by the caller.
    """

    def sample(
        self,
        slides: Iterable[SlideSpec],
        *,
        context: SamplingContext | None = None,
    ) -> Iterable[PatchRequest]:
        """Return requests assigned to the supplied worker context."""
