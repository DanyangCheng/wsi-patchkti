"""Sampling from precomputed patch requests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..types import PatchRequest, SamplingContext, SlideSpec


@dataclass(frozen=True, slots=True)
class IndexedSampler:
    """Enumerate or randomly draw from a precomputed request index."""

    requests: Sequence[PatchRequest]
    num_samples: int | None = None
    weights: Sequence[float] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        if not requests:
            raise ValueError("requests must not be empty")
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "requests", requests)
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
            for index, request in enumerate(self.requests):
                if context.owns(index):
                    yield request
            return
        probabilities = None
        if self.weights is not None:
            probabilities = np.asarray(self.weights, dtype=np.float64)
            probabilities /= probabilities.sum()
        for global_index in range(self.num_samples):
            if not context.owns(global_index):
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, context.epoch, global_index])
            )
            selected = int(rng.choice(len(self.requests), p=probabilities))
            yield self.requests[selected]
