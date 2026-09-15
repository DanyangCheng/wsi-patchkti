from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tifffile

from wsi_patchkit import PatchRequest, PatchStream, TiffReader, read_aligned_patch


def _write_slide(path: Path) -> np.ndarray:
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        rowsperstrip=2,
        resolution=(40_000, 40_000),  # 0.25 micrometres per pixel
        resolutionunit="CENTIMETER",
        metadata=None,
    )
    return image


def _write_pyramid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    level0 = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    level1 = level0[::2, ::2]
    with tifffile.TiffWriter(path) as writer:
        writer.write(
            level0,
            photometric="rgb",
            tile=(16, 16),
            subifds=1,
            resolution=(40_000, 40_000),
            resolutionunit="CENTIMETER",
            metadata=None,
        )
        writer.write(
            level1,
            photometric="rgb",
            tile=(16, 16),
            subfiletype=1,
            metadata=None,
        )
    return level0, level1


def test_tiff_reader_reports_mpp_and_reads_only_the_region(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    image = _write_slide(path)
    reader = TiffReader(cache_size=1)

    metadata = reader.metadata(path)
    region = reader.read_region(path, (2, 1), 0, (4, 3))

    assert metadata.dimensions == (10, 8)
    assert metadata.mpp == (0.25, 0.25)
    np.testing.assert_array_equal(region, image[1:4, 2:6])
    reader.close()


def test_level_location_uses_level_zero_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "pyramid.tif"
    _, level1 = _write_pyramid(path)
    reader = TiffReader()

    metadata = reader.metadata(path)
    region = reader.read_region(path, (2, 2), 1, (2, 2))

    assert [level.dimensions for level in metadata.levels] == [(8, 8), (4, 4)]
    assert metadata.levels[1].mpp == (0.5, 0.5)
    np.testing.assert_array_equal(region, level1[1:3, 1:3])
    reader.close()


def test_aligned_patch_selects_physical_level_and_pads_boundaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pyramid.tif"
    _, level1 = _write_pyramid(path)
    reader = TiffReader()
    request = PatchRequest(path, -1, -1, 3, 3, target_mpp=0.5)

    patch = read_aligned_patch(reader, request, interpolation="nearest")

    assert patch.shape == (3, 3, 3)
    assert np.all(patch[0] == 255)
    assert np.all(patch[:, 0] == 255)
    np.testing.assert_array_equal(patch[1:, 1:], level1[:2, :2])
    reader.close()


def test_patch_stream_reuses_metadata_and_materializes_requests(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    image = _write_slide(path)
    reader = TiffReader()
    requests = [
        PatchRequest(path, 0, 0, 2, 2, 0.25),
        PatchRequest(path, 2, 0, 2, 2, 0.25),
    ]

    patches = list(PatchStream(reader, requests))

    np.testing.assert_array_equal(patches[0].image, image[:2, :2])
    np.testing.assert_array_equal(patches[1].image, image[:2, 2:4])
    reader.close()


def test_tiff_reader_drops_handles_when_pickled(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    _write_slide(path)
    reader = TiffReader(cache_size=2)
    reader.metadata(path)
    reader.read_region(path, (0, 0), 0, (1, 1))

    restored = pickle.loads(pickle.dumps(reader))

    assert restored.cache_size == 2
    assert not restored._readers
    assert restored.metadata(path).dimensions == (10, 8)
    reader.close()
    restored.close()
