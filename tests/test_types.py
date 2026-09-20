from __future__ import annotations

import pytest

from wsi_patchkit import LevelInfo, PixelFormat, SlideMetadata, SlideSpec, as_mpp


def test_scalar_mpp_and_virtual_canvas() -> None:
    metadata = SlideMetadata(
        "slide.svs",
        (LevelInfo(0, (100, 80), (1.0, 1.0), (0.25, 0.25)),),
        mpp=(0.25, 0.25),
    )

    slide = SlideSpec.from_metadata(metadata, target_mpp=0.5)

    assert slide.canvas_size == (50, 40)
    assert slide.target_mpp == (0.5, 0.5)


@pytest.mark.parametrize("value", [0, -1, (0.5,), (0.5, 0)])
def test_invalid_mpp_is_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        as_mpp(value)  # type: ignore[arg-type]


def test_metadata_requires_consecutive_levels() -> None:
    with pytest.raises(ValueError, match="numbered from zero"):
        SlideMetadata(
            "slide.svs",
            (LevelInfo(1, (10, 10), (2.0, 2.0)),),
        )


def test_level_can_describe_reader_pixel_format() -> None:
    level = LevelInfo(
        0,
        (10, 10),
        (1.0, 1.0),
        pixel_format=PixelFormat("uint8", 3, color_model="rgb", photometric="rgb"),
    )

    assert level.pixel_format is not None
    assert level.pixel_format.dtype == "uint8"
    assert level.pixel_format.channels == 3
