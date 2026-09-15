"""Optional PyTorch iterable-dataset adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from ..io.base import SlideReader
from ..sampling.base import PatchSampler
from ..sampling.tissue import TissueFilter
from ..stream import PatchStream
from ..types import Patch, SamplingContext, SlideSpec


class WSIPatchIterableDataset(IterableDataset[Mapping[str, Any]]):
    """Materialize sampler requests with one reader per DataLoader worker."""

    def __init__(
        self,
        slides: Sequence[SlideSpec],
        sampler: PatchSampler,
        *,
        reader_factory: Callable[[], SlideReader],
        tissue_filter: TissueFilter | None = None,
        transform: Callable[[Patch], Mapping[str, Any]] | None = None,
        epoch: int = 0,
    ) -> None:
        super().__init__()
        if not slides:
            raise ValueError("slides must not be empty")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.slides = tuple(slides)
        self.sampler = sampler
        self.reader_factory = reader_factory
        self.tissue_filter = tissue_filter
        self.transform = transform
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic random stream for the next iteration."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _context(self) -> SamplingContext:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        distributed = dist.is_available() and dist.is_initialized()
        return SamplingContext(
            epoch=self.epoch,
            rank=dist.get_rank() if distributed else 0,
            world_size=dist.get_world_size() if distributed else 1,
            worker_id=worker_id,
            num_workers=num_workers,
        )

    @staticmethod
    def _default_item(patch: Patch) -> Mapping[str, Any]:
        image = np.asarray(patch.image)
        if image.ndim != 3:
            raise ValueError(f"expected an HWC patch, got {image.shape}")
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        return {"image": tensor, "request": patch.request}

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        requests = self.sampler.sample(self.slides, context=self._context())
        if self.tissue_filter is not None:
            requests = self.tissue_filter.filter(requests)
        reader = self.reader_factory()
        try:
            for patch in PatchStream(reader, requests):
                yield (
                    self.transform(patch)
                    if self.transform is not None
                    else self._default_item(patch)
                )
        finally:
            reader.close()
