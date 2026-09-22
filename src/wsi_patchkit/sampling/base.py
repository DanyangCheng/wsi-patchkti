"""Sampling interfaces and shared helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..types import PatchRequest, SamplingContext, SlideSpec


@runtime_checkable
class PatchRequestSource(Protocol):
    """A lazily addressable source of precomputed patch requests.

    Storage and serialization are application concerns; sources may wrap an
    in-memory sequence, memory-mapped array, or database-backed index.
    """

    def __len__(self) -> int:
        """Return the number of requests."""

    def __getitem__(self, index: int) -> PatchRequest:
        """Return one request by zero-based position."""


class PatchRequestSampler(Protocol):
    """Protocol implemented by slide-aware patch-request samplers.

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
