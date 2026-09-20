"""Sampling from precomputed patch requests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..types import PatchRequest, SamplingContext, SlideSpec
from .base import PatchRequestSource


@dataclass(frozen=True, slots=True)
class IndexedSampler:
    """Enumerate or randomly draw from a precomputed request index."""

    requests: PatchRequestSource
    num_samples: int | None = None
    weights: Sequence[float] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        requests = self.requests
        if len(requests) < 1:
            raise ValueError("requests must not be empty")
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.weights is not None:
            weights = tuple(float(item) for item in self.weights)
            if len(weights) != len(requests):
                raise ValueError("weights must have one value per request")
            if any(not np.isfinite(item) or item < 0 for item in weights):
                raise ValueError("weights must be finite and non-negative")
            if not any(item > 0 for item in weights):
                raise ValueError("at least one weight must be positive")
            object.__setattr__(self, "weights", weights)

    def sample(
        self,
        slides: Iterable[SlideSpec] = (),
        *,
        context: SamplingContext | None = None,
    ) -> Iterator[PatchRequest]:
        del slides
        context = context or SamplingContext()
        if self.num_samples is None:
            total = len(self.requests)
            for virtual_index in context.indices(total):
                yield self.requests[context.source_index(virtual_index, total)]
            return
        probabilities = None
        if self.weights is not None:
            probabilities = np.asarray(self.weights, dtype=np.float64)
            probabilities /= probabilities.sum()
        for virtual_index in context.indices(self.num_samples):
            global_index = context.source_index(virtual_index, self.num_samples)
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, context.epoch, global_index])
            )
            selected = int(rng.choice(len(self.requests), p=probabilities))
            yield self.requests[selected]
