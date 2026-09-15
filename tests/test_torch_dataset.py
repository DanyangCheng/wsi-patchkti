from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

torch = pytest.importorskip("torch")

from wsi_patchkit import GridSampler, SlideSpec  # noqa: E402
from wsi_patchkit.io import TiffReader  # noqa: E402
from wsi_patchkit.torch import WSIPatchIterableDataset  # noqa: E402


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
