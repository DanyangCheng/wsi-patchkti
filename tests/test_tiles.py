from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from wsi_patchkit import (
    LevelInfo,
    SlideMetadata,
    TileRenderer,
    choose_level_for_downsample,
    iiif_scale_factors,
)


class RecordingReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.level0 = np.zeros((8, 10, 3), dtype=np.uint8)
        self.level0[..., 0] = np.arange(10, dtype=np.uint8)
        self.level0[..., 1] = np.arange(8, dtype=np.uint8)[:, None]
        self.level1 = self.level0[::2, ::2]
        self.calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
        self.closed = False

    def metadata(self, path: str | Path, *, source_mpp=None) -> SlideMetadata:
        assert Path(path) == self.path
        mpp = (0.25, 0.25) if source_mpp is None else source_mpp
        return SlideMetadata(
            path,
            (
                LevelInfo(0, (10, 8), (1, 1), mpp),
                LevelInfo(1, (5, 4), (2, 2), (mpp[0] * 2, mpp[1] * 2)),
            ),
            mpp=mpp,
        )

    def read_region(self, path, location, level, size):
        self.calls.append((location, level, size))
        array = self.level0 if level == 0 else self.level1
        downsample = 1 if level == 0 else 2
        x, y = location[0] // downsample, location[1] // downsample
        width, height = size
        return array[y : y + height, x : x + width]

    def close(self) -> None:
        self.closed = True


def test_iiif_scale_factors_cover_the_smallest_view() -> None:
    assert iiif_scale_factors((1000, 513), tile_size=256) == (1, 2, 4)
    assert iiif_scale_factors((128, 64), tile_size=256) == (1,)


def test_level_selection_prefers_finer_native_data() -> None:
    metadata = SlideMetadata(
        "slide.tif",
        (
            LevelInfo(0, (100, 80), (1, 1)),
            LevelInfo(1, (25, 20), (4, 4)),
            LevelInfo(2, (10, 8), (10, 10)),
        ),
    )

    assert choose_level_for_downsample(metadata, 8).level == 1
    assert choose_level_for_downsample(metadata, 12).level == 2


def test_tile_renderer_selects_level_encodes_and_caches(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    path.touch()
    reader = RecordingReader(path)
    renderer = TileRenderer(reader, cache_size=2)

    first = renderer.render_region(path, (0, 0, 10, 8), (5, 4), image_format="png")
    second = renderer.render_region(path, (0, 0, 10, 8), (5, 4), image_format="png")

    assert first is second
    assert first.media_type == "image/png"
    assert (first.width, first.height) == (5, 4)
    assert reader.calls == [((0, 0), 1, (5, 4))]
    decoded = np.asarray(Image.open(BytesIO(first.content)))
    np.testing.assert_array_equal(decoded, reader.level1)
    renderer.close()
    assert reader.closed


def test_tile_renderer_crops_edge_regions(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    path.touch()
    reader = RecordingReader(path)
    renderer = TileRenderer(reader)

    result = renderer.render_region(path, (8, 6, 4, 4), (4, 4), image_format="png")

    assert (result.width, result.height) == (2, 2)
    assert reader.calls == [((8, 6), 0, (2, 2))]
    renderer.close()

