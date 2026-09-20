from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from wsi_patchkit import GridSampler, Patch, RandomSampler, SlideSpec  # noqa: E402
from wsi_patchkit.io import TiffReader  # noqa: E402
from wsi_patchkit.torch import WSIPatchIterableDataset  # noqa: E402


def _coordinates(patch: Patch) -> dict[str, object]:
    request = patch.request
    return {"coordinates": torch.tensor((request.x, request.y))}


def test_torch_dataset_returns_chw_float_tensor(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    tifffile.imwrite(
        path,
        np.full((4, 4, 3), 128, dtype=np.uint8),
        photometric="rgb",
        resolution=(20_000, 20_000),
        resolutionunit="CENTIMETER",
        metadata=None,
    )
    slide = SlideSpec(path, canvas_size=(4, 4), target_mpp=0.5)
    dataset = WSIPatchIterableDataset(
        [slide],
        GridSampler(2),
        reader_factory=TiffReader,
    )

    item = next(iter(dataset))

    assert item["image"].shape == (3, 2, 2)
    assert item["image"].dtype == torch.float32
    torch.testing.assert_close(item["image"], torch.full((3, 2, 2), 128 / 255))


def test_dataset_epoch_is_visible_to_persistent_workers(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    tifffile.imwrite(
        path,
        np.full((16, 16, 3), 128, dtype=np.uint8),
        photometric="rgb",
        resolution=(20_000, 20_000),
        resolutionunit="CENTIMETER",
        metadata=None,
    )
    dataset = WSIPatchIterableDataset(
        [SlideSpec(path, canvas_size=(16, 16), target_mpp=0.5)],
        RandomSampler(num_samples=12, patch_size=2, seed=7),
        reader_factory=TiffReader,
        transform=_coordinates,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
    )
    try:
        first = [tuple(item["coordinates"].tolist()) for item in loader]
        dataset.set_epoch(1)
        second = [tuple(item["coordinates"].tolist()) for item in loader]
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    assert first != second


def test_dataset_checkpoint_state_restores_epoch_and_cursor(tmp_path: Path) -> None:
    dataset = WSIPatchIterableDataset(
        [SlideSpec(tmp_path / "slide.tif", canvas_size=(4, 4), target_mpp=0.5)],
        GridSampler(2),
        reader_factory=TiffReader,
        epoch=2,
        start_index=7,
    )

    state = dataset.state_dict()
    dataset.set_epoch(3)
    dataset.load_state_dict(state)

    assert dataset.epoch == 2
    assert dataset.start_index == 7
