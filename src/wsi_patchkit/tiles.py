"""Protocol-neutral rendering of browser-friendly WSI image tiles."""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .io.base import SlideReader
from .types import MPP, LevelInfo, SlideMetadata, as_mpp

ImageFormat = Literal["jpg", "png"]


@dataclass(frozen=True, slots=True)
class EncodedImage:
    """An encoded image response and its cache metadata."""

    content: bytes
    media_type: str
    width: int
    height: int
    etag: str


def iiif_scale_factors(
    dimensions: tuple[int, int],
    tile_size: int = 256,
) -> tuple[int, ...]:
    """Return a power-of-two virtual pyramid covering an image."""
    width, height = map(int, dimensions)
    if width < 1 or height < 1 or tile_size < 1:
        raise ValueError("dimensions and tile_size must be positive")
    factors = [1]
    while math.ceil(width / factors[-1]) > tile_size or math.ceil(
        height / factors[-1]
    ) > tile_size:
        factors.append(factors[-1] * 2)
    return tuple(factors)


def choose_level_for_downsample(
    metadata: SlideMetadata,
    requested: float | Sequence[float],
) -> LevelInfo:
    """Choose the coarsest native level no coarser than the requested view.

    Reading a slightly finer level and reducing it avoids visible blur caused by
    upscaling a coarser native WSI level.
    """
    requested_pair = as_mpp(requested, name="requested_downsample")
    eligible = [
        level
        for level in metadata.levels
        if level.downsample[0] <= requested_pair[0]
        and level.downsample[1] <= requested_pair[1]
    ]
    if not eligible:
        return metadata.levels[0]
    return max(eligible, key=lambda item: math.prod(item.downsample))


def _as_rgb(array: NDArray[np.generic]) -> NDArray[np.uint8]:
    if array.dtype != np.uint8:
        raise ValueError("web tiles currently require an 8-bit WSI")
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"reader returned unsupported shape {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        alpha = array[..., 3:4].astype(np.uint32)
        array = (
            array[..., :3].astype(np.uint32) * alpha
            + 255 * (255 - alpha)
            + 127
        ) // 255
    elif array.shape[2] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"reader returned unsupported channel count {array.shape[2]}")
    return np.asarray(array, dtype=np.uint8)


class TileRenderer:
    """Render arbitrary level-0 regions with native-level selection and caching.

    Reader calls are serialized because common TIFF and OpenSlide handles are not
    safe to seek from multiple request threads at the same time. Applications can
    scale across processes or construct multiple renderer instances.
    """

    def __init__(
        self,
        reader: SlideReader,
        *,
        cache_size: int = 512,
        jpeg_quality: int = 85,
        max_output_pixels: int = 16_777_216,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if max_output_pixels < 1:
            raise ValueError("max_output_pixels must be positive")
        self.reader = reader
        self.cache_size = int(cache_size)
        self.jpeg_quality = int(jpeg_quality)
        self.max_output_pixels = int(max_output_pixels)
        self._cache: OrderedDict[tuple[object, ...], EncodedImage] = OrderedDict()
        self._lock = threading.RLock()

    def metadata(
        self,
        path: str | Path,
        *,
        source_mpp: float | Sequence[float] | None = None,
    ) -> SlideMetadata:
        with self._lock:
            return self.reader.metadata(path, source_mpp=source_mpp)

    @staticmethod
    def _fingerprint(path: str | Path) -> tuple[str, int, int]:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        return str(resolved), stat.st_size, stat.st_mtime_ns

    def render_region(
        self,
        path: str | Path,
        region: tuple[int, int, int, int],
        output_size: tuple[int, int],
        *,
        image_format: ImageFormat = "jpg",
        source_mpp: float | Sequence[float] | None = None,
    ) -> EncodedImage:
        """Render a level-0 ``(x, y, width, height)`` region."""
        x, y, width, height = map(int, region)
        output_width, output_height = map(int, output_size)
        if x < 0 or y < 0 or width < 1 or height < 1:
            raise ValueError("region must contain non-negative coordinates and size")
        if output_width < 1 or output_height < 1:
            raise ValueError("output size must be positive")
        if output_width * output_height > self.max_output_pixels:
            raise ValueError("requested output exceeds max_output_pixels")
        if image_format not in ("jpg", "png"):
            raise ValueError("image_format must be 'jpg' or 'png'")

        source_mpp_pair: MPP | None = (
            None if source_mpp is None else as_mpp(source_mpp, name="source_mpp")
        )
        fingerprint = self._fingerprint(path)
        key: tuple[object, ...] = (
            fingerprint,
            source_mpp_pair,
            x,
            y,
            width,
            height,
            output_width,
            output_height,
            image_format,
            self.jpeg_quality,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

            metadata = self.reader.metadata(path, source_mpp=source_mpp_pair)
            full_width, full_height = metadata.dimensions
            if x >= full_width or y >= full_height:
                raise ValueError("region starts outside the image")
            clipped_width = min(width, full_width - x)
            clipped_height = min(height, full_height - y)
            rendered_width = max(1, round(output_width * clipped_width / width))
            rendered_height = max(1, round(output_height * clipped_height / height))
            requested_downsample = (
                clipped_width / rendered_width,
                clipped_height / rendered_height,
            )
            level = choose_level_for_downsample(metadata, requested_downsample)

            lx0 = max(0, math.floor(x / level.downsample[0]))
            ly0 = max(0, math.floor(y / level.downsample[1]))
            lx1 = min(
                level.dimensions[0],
                math.ceil((x + clipped_width) / level.downsample[0]),
            )
            ly1 = min(
                level.dimensions[1],
                math.ceil((y + clipped_height) / level.downsample[1]),
            )
            level_zero_location = (
                round(lx0 * level.downsample[0]),
                round(ly0 * level.downsample[1]),
            )
            array = self.reader.read_region(
                path,
                level_zero_location,
                level.level,
                (lx1 - lx0, ly1 - ly0),
            )
            image = Image.fromarray(_as_rgb(array), "RGB")
            extent = (
                (x / level.downsample[0]) - lx0,
                (y / level.downsample[1]) - ly0,
                ((x + clipped_width) / level.downsample[0]) - lx0,
                ((y + clipped_height) / level.downsample[1]) - ly0,
            )
            image = image.transform(
                (rendered_width, rendered_height),
                Image.Transform.EXTENT,
                extent,
                resample=Image.Resampling.BILINEAR,
            )

            buffer = BytesIO()
            if image_format == "jpg":
                image.save(buffer, format="JPEG", quality=self.jpeg_quality)
                media_type = "image/jpeg"
            else:
                image.save(buffer, format="PNG")
                media_type = "image/png"
            content = buffer.getvalue()
            result = EncodedImage(
                content,
                media_type,
                rendered_width,
                rendered_height,
                f'"{hashlib.sha256(content).hexdigest()}"',
            )
            if self.cache_size:
                self._cache[key] = result
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return result

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self.reader.close()
