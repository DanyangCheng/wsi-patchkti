"""Optional PyTorch iterable-dataset adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from ..geometry import Interpolation
from ..io.base import SlideReader
from ..sampling.base import PatchRequestSampler
from ..sampling.tissue import TissueFilter
from ..stream import PatchStream
from ..types import Patch, SamplingContext, SlideSpec


class WSIPatchIterableDataset(IterableDataset[Mapping[str, Any]]):
    """Materialize patch-request samples with one reader per DataLoader worker.

    By default, the final global sampling round is padded so every distributed
    rank and DataLoader worker receives an equal number of requests. Padding
    repeats deterministic requests from the start of the epoch. This prevents a
    short rank from ending a DDP iteration early.
    """

    def __init__(
        self,
        slides: Sequence[SlideSpec],
        request_sampler: PatchRequestSampler,
        *,
        reader_factory: Callable[[], SlideReader],
        tissue_filter: TissueFilter | None = None,
        transform: Callable[[Patch], Mapping[str, Any]] | None = None,
        interpolation: Interpolation = "bilinear",
        epoch: int = 0,
        even_shards: Literal["pad", "drop", "none"] = "pad",
        start_index: int = 0,
    ) -> None:
        super().__init__()
        if not slides:
            raise ValueError("slides must not be empty")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if interpolation not in ("nearest", "bilinear", "area"):
            raise ValueError(
                "interpolation must be 'nearest', 'bilinear', or 'area'"
            )
        if even_shards not in ("pad", "drop", "none"):
            raise ValueError("even_shards must be 'pad', 'drop', or 'none'")
        if tissue_filter is not None and even_shards != "none":
            raise ValueError(
                "TissueFilter runs after sharding and cannot guarantee equal "
                "DDP lengths; use even_shards='none' or a tissue-aware "
                "patch-request sampler"
            )
        self.slides = tuple(slides)
        self.request_sampler = request_sampler
        self.reader_factory = reader_factory
        self.tissue_filter = tissue_filter
        self.transform = transform
        self.interpolation = interpolation
        self.even_shards = even_shards
        # A shared tensor keeps set_epoch visible to persistent DataLoader
        # workers under both fork and spawn multiprocessing start methods.
        self._sampling_state = torch.tensor(
            [epoch, start_index], dtype=torch.int64
        ).share_memory_()

    @property
    def epoch(self) -> int:
        """The epoch currently visible to all workers."""
        return int(self._sampling_state[0].item())

    @property
    def start_index(self) -> int:
        """The next virtual global request-sampler index used on iteration."""
        return int(self._sampling_state[1].item())

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        """Select a deterministic epoch and optional resume cursor.

        Call this before creating a new DataLoader iterator. ``start_index`` is
        a virtual global index as defined by :class:`SamplingContext`; callers
        should checkpoint it at a synchronized training-step boundary.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        self._sampling_state[0] = epoch
        self._sampling_state[1] = start_index

    def set_start_index(self, start_index: int) -> None:
        """Set the checkpoint resume cursor without changing the epoch."""
        self.set_epoch(self.epoch, start_index=start_index)

    def state_dict(self) -> dict[str, int | str]:
        """Return checkpointable epoch and request-sampler cursor state.

        The caller owns advancing ``start_index`` while consuming batches. This
        is deliberate: DataLoader prefetching means a worker-local iterator
        cannot reliably identify the last optimizer step committed by training.
        """
        return {
            "epoch": self.epoch,
            "start_index": self.start_index,
            "even_shards": self.even_shards,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore state returned by :meth:`state_dict`."""
        try:
            epoch = state["epoch"]
            start_index = state["start_index"]
            even_shards = state["even_shards"]
        except KeyError as error:
            raise ValueError(f"dataset state is missing {error.args[0]!r}") from error
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("dataset state epoch must be an integer")
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise ValueError("dataset state start_index must be an integer")
        if even_shards != self.even_shards:
            raise ValueError(
                "dataset state even_shards does not match this dataset instance"
            )
        self.set_epoch(epoch, start_index=start_index)

    def _context(self) -> SamplingContext:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        distributed = dist.is_available() and dist.is_initialized()
        shard_policy = {"pad": "pad", "drop": "drop", "none": "uneven"}[
            self.even_shards
        ]
        return SamplingContext(
            epoch=self.epoch,
            rank=dist.get_rank() if distributed else 0,
            world_size=dist.get_world_size() if distributed else 1,
            worker_id=worker_id,
            num_workers=num_workers,
            shard_policy=shard_policy,
            start_index=self.start_index,
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
        requests = self.request_sampler.sample(self.slides, context=self._context())
        if self.tissue_filter is not None:
            requests = self.tissue_filter.filter(requests)
        reader = self.reader_factory()
        try:
            for patch in PatchStream(
                reader,
                requests,
                interpolation=self.interpolation,
            ):
                yield (
                    self.transform(patch)
                    if self.transform is not None
                    else self._default_item(patch)
                )
        finally:
            reader.close()
