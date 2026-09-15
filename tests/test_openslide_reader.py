from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

openslide = pytest.importorskip("openslide")

from wsi_patchkit.io import OpenSlideReader  # noqa: E402


def test_openslide_reader_uses_level_zero_location_contract(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(16, 16),
        compression="deflate",
        resolution=(40_000, 40_000),
        resolutionunit="CENTIMETER",
        metadata=None,
    )
    reader = OpenSlideReader()

    metadata = reader.metadata(path)
    region = reader.read_region(path, (2, 3), 0, (4, 5))

    assert metadata.dimensions == (32, 32)
    assert metadata.mpp == pytest.approx((0.25, 0.25))
    np.testing.assert_array_equal(region, image[3:8, 2:6])
    reader.close()
