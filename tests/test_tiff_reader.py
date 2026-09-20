from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tifffile

from wsi_patchkit import (
    LevelInfo,
    PatchRequest,
    PatchStream,
    SlideMetadata,
    TiffReader,
    choose_level,
    read_aligned_patch,
    read_aligned_patch_result,
)


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
    assert metadata.levels[0].pixel_format is not None
    assert metadata.levels[0].pixel_format.color_model == "rgb"
    np.testing.assert_array_equal(region, image[1:4, 2:6])
    reader.close()


def test_tiff_reader_handles_bigtiff_metadata_and_regions(tmp_path: Path) -> None:
    path = tmp_path / "slide.btf"
    image = np.arange(6 * 6, dtype=np.uint16).reshape(6, 6)
    tifffile.imwrite(
        path,
        image,
        bigtiff=True,
        rowsperstrip=2,
        resolution=(20_000, 20_000),
        resolutionunit="CENTIMETER",
        metadata=None,
    )
    reader = TiffReader()

    metadata = reader.metadata(path)
    region = reader.read_region(path, (1, 2), 0, (3, 2))

    assert metadata.levels[0].pixel_format is not None
    assert metadata.levels[0].pixel_format.dtype == "uint16"
    assert metadata.levels[0].pixel_format.color_model == "gray"
    np.testing.assert_array_equal(region[..., 0], image[2:4, 1:4])
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


def test_aligned_read_round_trips_non_integral_pyramid_coordinates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "non-integral-pyramid.tif"
    level0 = np.zeros((10, 10, 3), dtype=np.uint8)
    level1 = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
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
    reader = TiffReader()
    metadata = reader.metadata(path)
    request = PatchRequest(
        path,
        1,
        1,
        1,
        1,
        target_mpp=metadata.levels[1].mpp,
    )

    result = read_aligned_patch_result(
        reader,
        request,
        metadata=metadata,
        level=1,
        interpolation="nearest",
    )

    assert result.plan is not None
    assert result.plan.level_zero_location == (4, 4)
    np.testing.assert_array_equal(result.image, level1[1:2, 1:2])
    reader.close()


def test_area_interpolation_averages_rgb_when_downsampling(tmp_path: Path) -> None:
    path = tmp_path / "checkerboard.tif"
    checkerboard = (
        (np.indices((4, 4)).sum(axis=0) % 2 * 255)
        .astype(np.uint8)[..., None]
        .repeat(3, axis=2)
    )
    tifffile.imwrite(
        path,
        checkerboard,
        photometric="rgb",
        resolution=(40_000, 40_000),
        resolutionunit="CENTIMETER",
        metadata=None,
    )
    reader = TiffReader()

    patch = read_aligned_patch(
        reader,
        PatchRequest(path, 0, 0, 2, 2, target_mpp=0.5),
        interpolation="area",
    )

    np.testing.assert_array_equal(patch, np.full((2, 2, 3), 128, dtype=np.uint8))
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


def test_patch_read_result_exposes_geometric_coverage(tmp_path: Path) -> None:
    path = tmp_path / "pyramid.tif"
    _, level1 = _write_pyramid(path)
    reader = TiffReader()
    request = PatchRequest(path, -1, -1, 3, 3, target_mpp=0.5)

    result = read_aligned_patch_result(reader, request, interpolation="nearest")

    assert result.plan is not None
    assert result.valid_mask.dtype == bool
    np.testing.assert_array_equal(
        result.valid_mask,
        np.array([[False, False, False], [False, True, True], [False, True, True]]),
    )
    np.testing.assert_array_equal(result.image[1:, 1:], level1[:2, :2])
    reader.close()


def test_patch_read_result_marks_fully_outside_request_invalid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "slide.tif"
    _write_slide(path)
    reader = TiffReader()

    result = read_aligned_patch_result(
        reader,
        PatchRequest(path, 100, 100, 2, 2, target_mpp=0.25),
    )

    assert result.plan is None
    assert not result.valid_mask.any()
    assert np.all(result.image == 255)
    reader.close()


def test_finer_level_policy_avoids_upscaling_a_coarser_level() -> None:
    metadata = SlideMetadata(
        "slide.tif",
        (
            LevelInfo(0, (100, 100), (1, 1), (0.25, 0.25)),
            LevelInfo(1, (50, 50), (2, 2), (0.5, 0.5)),
            LevelInfo(2, (25, 25), (4, 4), (1.0, 1.0)),
        ),
        mpp=(0.25, 0.25),
    )

    assert choose_level(metadata, (0.75, 0.75)).level == 2
    assert choose_level(metadata, (0.75, 0.75), policy="finer").level == 1


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
