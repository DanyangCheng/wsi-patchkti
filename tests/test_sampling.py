from __future__ import annotations

import numpy as np
import pytest

from wsi_patchkit import (
    GridSampler,
    IndexedSampler,
    PatchRequest,
    RandomSampler,
    SamplingContext,
    SlideSpec,
    TissueFilter,
    TissueMask,
    axis_positions,
)


def _slide(name: str = "slide.tif") -> SlideSpec:
    return SlideSpec(name, canvas_size=(10, 8), target_mpp=0.5)


def test_grid_aligns_the_last_window_to_each_edge() -> None:
    requests = list(GridSampler(patch_size=4, stride=3).sample([_slide()]))

    assert axis_positions(10, 4, 3) == (0, 3, 6)
    assert axis_positions(8, 4, 3) == (0, 3, 4)
    assert [(item.x, item.y) for item in requests] == [
        (0, 0),
        (3, 0),
        (6, 0),
        (0, 3),
        (3, 3),
        (6, 3),
        (0, 4),
        (3, 4),
        (6, 4),
    ]


def test_drop_policy_omits_an_incomplete_small_slide() -> None:
    slide = SlideSpec("small.tif", canvas_size=(2, 2), target_mpp=0.5)

    assert list(GridSampler(4, edge="drop").sample([slide])) == []


def test_aligned_grid_rejects_gaps() -> None:
    with pytest.raises(ValueError, match="stride must not exceed"):
        GridSampler(patch_size=4, stride=5)


def test_grid_workers_are_disjoint_and_complete() -> None:
    sampler = GridSampler(patch_size=4, stride=3)
    full = list(sampler.sample([_slide()]))
    worker0 = list(
        sampler.sample(
            [_slide()],
            context=SamplingContext(worker_id=0, num_workers=2),
        )
    )
    worker1 = list(
        sampler.sample(
            [_slide()],
            context=SamplingContext(worker_id=1, num_workers=2),
        )
    )

    assert worker0 == full[::2]
    assert worker1 == full[1::2]
    assert set(worker0).isdisjoint(worker1)


def test_random_sampling_is_reproducible_and_epoch_sensitive() -> None:
    slides = [_slide("a.tif"), _slide("b.tif")]
    sampler = RandomSampler(num_samples=30, patch_size=4, seed=19)

    first = list(sampler.sample(slides, context=SamplingContext(epoch=2)))
    repeated = list(sampler.sample(slides, context=SamplingContext(epoch=2)))
    next_epoch = list(sampler.sample(slides, context=SamplingContext(epoch=3)))

    assert first == repeated
    assert first != next_epoch
    assert all(0 <= item.x <= 6 and 0 <= item.y <= 4 for item in first)


def test_random_worker_shards_rebuild_the_global_sample_set() -> None:
    sampler = RandomSampler(num_samples=17, patch_size=3, seed=2)
    full = list(sampler.sample([_slide()]))
    workers = [
        list(
            sampler.sample(
                [_slide()],
                context=SamplingContext(worker_id=worker, num_workers=3),
            )
        )
        for worker in range(3)
    ]

    rebuilt = [
        request
        for index in range(17)
        for request in workers[index % 3][index // 3 : index // 3 + 1]
    ]
    assert rebuilt == full


def test_indexed_sampling_validates_and_shards_requests() -> None:
    requests = tuple(PatchRequest("slide.tif", x, 0, 2, 2, 0.5) for x in range(5))
    sampler = IndexedSampler(requests)

    assert list(
        sampler.sample(context=SamplingContext(worker_id=1, num_workers=2))
    ) == list(requests[1::2])
    with pytest.raises(ValueError, match="one value per request"):
        IndexedSampler(requests, weights=(1.0,))


def test_tissue_filter_maps_a_low_resolution_mask_to_the_canvas() -> None:
    mask = TissueMask(
        np.array([[1, 0], [1, 0]], dtype=np.uint8),
        canvas_size=(10, 8),
    )
    requests = list(GridSampler(patch_size=(5, 4), stride=(5, 4)).sample([_slide()]))

    selected = list(
        TissueFilter({"slide.tif": mask}, minimum_fraction=0.5).filter(requests)
    )

    assert [(item.x, item.y) for item in selected] == [(0, 0), (0, 4)]
