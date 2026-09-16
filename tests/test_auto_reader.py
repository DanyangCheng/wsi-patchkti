from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

import wsi_patchkit.io.auto as auto_module
from wsi_patchkit.io import AutoSlideReader


def _write_tiff(path: Path, *, description: str | None = None) -> None:
    tifffile.imwrite(
        path,
        np.zeros((16, 16, 3), dtype=np.uint8),
        photometric="rgb",
        tile=(16, 16),
        description=description,
        metadata=None,
    )


def test_auto_reader_keeps_ordinary_tiff_on_tifffile_backend(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ordinary.tiff"
    _write_tiff(path)
    reader = AutoSlideReader()

    metadata = reader.metadata(path)
    region = reader.read_region(path, (0, 0), 0, (4, 3))

    assert metadata.vendor == "generic-tiff"
    assert region.shape == (3, 4, 3)
    assert reader._openslide is None
    reader.close()


def test_auto_reader_routes_aperio_tiff_to_openslide(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "aperio.tiff"
    _write_tiff(path, description="Aperio Image|MPP =0.25")
    calls: list[str] = []
    cache_sizes: list[int] = []

    class FakeOpenSlideReader:
        def __init__(self, cache_size: int = 4) -> None:
            calls.append("init")
            cache_sizes.append(cache_size)

        def metadata(self, value, *, source_mpp=None):
            calls.append("metadata")
            return "openslide-metadata"

        def read_region(self, value, location, level, size):
            calls.append("read_region")
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(auto_module, "OpenSlideReader", FakeOpenSlideReader)
    reader = AutoSlideReader(cache_size=2)

    assert reader.metadata(path) == "openslide-metadata"
    assert reader.read_region(path, (0, 0), 0, (4, 3)).shape == (3, 4, 3)
    assert calls == ["init", "metadata", "read_region"]
    assert cache_sizes == [2]
    assert reader._tiff_routes == {str(path.resolve()): True}
    reader.close()
