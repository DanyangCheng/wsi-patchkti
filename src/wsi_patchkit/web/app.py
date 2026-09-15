"""Optional FastAPI application serving IIIF tiles and the WSI viewer."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
except ImportError as error:  # pragma: no cover - exercised without the web extra
    raise ImportError(
        "The WSI viewer requires the 'web' extra: uv sync --extra web"
    ) from error

from ..io import AutoSlideReader, SlideReader
from ..tiles import TileRenderer, iiif_scale_factors
from ..types import SlideMetadata
from .registry import SlideRegistry, SlideSource
from .workers import TileWorkerPool

_STATIC_DIR = Path(__file__).with_name("static")
_LOGGER = logging.getLogger(__name__)
_STATIC_ASSETS = {
    "app.js": (_STATIC_DIR / "app.js").read_bytes(),
    "styles.css": (_STATIC_DIR / "styles.css").read_bytes(),
}
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _public_metadata(slide_id: str, metadata: SlideMetadata) -> dict[str, object]:
    return {
        "id": slide_id,
        "width": metadata.dimensions[0],
        "height": metadata.dimensions[1],
        "mpp": metadata.mpp,
        "vendor": metadata.vendor,
        "levels": [
            {
                "level": level.level,
                "width": level.dimensions[0],
                "height": level.dimensions[1],
                "downsample": level.downsample,
                "mpp": level.mpp,
            }
            for level in metadata.levels
        ],
    }


def _parse_region(value: str, dimensions: tuple[int, int]) -> tuple[int, int, int, int]:
    if value == "full":
        return 0, 0, dimensions[0], dimensions[1]
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError("region must be 'full' or x,y,width,height") from error
    if len(parts) != 4:
        raise ValueError("region must be 'full' or x,y,width,height")
    x, y, width, height = parts
    if x < 0 or y < 0 or width < 1 or height < 1:
        raise ValueError("region coordinates must be non-negative and sized")
    return x, y, width, height


def _parse_size(value: str, region: tuple[int, int, int, int]) -> tuple[int, int]:
    region_width, region_height = region[2:]
    if value == "max":
        return region_width, region_height
    parts = value.split(",")
    if len(parts) != 2 or (not parts[0] and not parts[1]):
        raise ValueError("size must be 'max', width, ,height, or width,height")
    try:
        width = int(parts[0]) if parts[0] else None
        height = int(parts[1]) if parts[1] else None
    except ValueError as error:
        raise ValueError("size values must be integers") from error
    if width is not None and width < 1 or height is not None and height < 1:
        raise ValueError("size values must be positive")
    if width is None:
        assert height is not None
        width = max(1, round(height * region_width / region_height))
    if height is None:
        height = max(1, round(width * region_height / region_width))
    return width, height


def create_app(
    slides: Mapping[str, SlideSource | str | Path],
    *,
    reader: SlideReader | None = None,
    reader_factory: Callable[[], SlideReader] | None = None,
    reader_pool_size: int = 4,
    tile_workers: int | None = None,
    tile_size: int = 256,
    cache_size: int = 512,
    jpeg_quality: int = 85,
    max_output_pixels: int = 16_777_216,
    cache_control: str = "private, max-age=3600",
) -> FastAPI:
    """Create a self-contained WSI viewer for an explicit slide registry."""
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if reader is not None and reader_factory is not None:
        raise ValueError("provide reader or reader_factory, not both")
    registry = SlideRegistry(slides)
    if reader is not None:
        renderer = TileRenderer(
            reader,
            cache_size=cache_size,
            jpeg_quality=jpeg_quality,
            max_output_pixels=max_output_pixels,
        )
    else:
        renderer = TileRenderer(
            reader_factory=reader_factory or AutoSlideReader,
            reader_pool_size=reader_pool_size,
            cache_size=cache_size,
            jpeg_quality=jpeg_quality,
            max_output_pixels=max_output_pixels,
        )
    worker_count = reader_pool_size if tile_workers is None else tile_workers
    workers = TileWorkerPool(worker_count)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        workers.close()
        renderer.close()

    app = FastAPI(title="wsi-patchkit viewer", lifespan=lifespan)
    app.state.slide_registry = registry
    app.state.tile_renderer = renderer
    app.state.tile_workers = workers

    def source_for(slide_id: str) -> SlideSource:
        try:
            return registry[slide_id]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown slide") from error

    def metadata_for(slide_id: str) -> tuple[SlideSource, SlideMetadata]:
        source = source_for(slide_id)
        try:
            metadata = renderer.metadata(source.path, source_mpp=source.source_mpp)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            _LOGGER.exception("Unable to read registered slide %s", slide_id)
            raise HTTPException(
                status_code=422,
                detail="registered slide could not be opened",
            ) from error
        return source, metadata

    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/static/{asset_name}", include_in_schema=False)
    async def static_asset(asset_name: str) -> Response:
        try:
            content = _STATIC_ASSETS[asset_name]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown asset") from error
        media_type = "text/javascript" if asset_name.endswith(".js") else "text/css"
        return Response(
            content,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/api/slides", name="list_slides")
    async def list_slides() -> list[dict[str, object]]:
        records = await asyncio.gather(
            *(workers.run(metadata_for, slide_id) for slide_id in registry)
        )
        return [
            _public_metadata(slide_id, record[1])
            for slide_id, record in zip(registry, records, strict=True)
        ]

    @app.get("/api/slides/{slide_id}", name="slide_metadata")
    async def slide_metadata(slide_id: str) -> dict[str, object]:
        _, metadata = await workers.run(metadata_for, slide_id)
        return _public_metadata(slide_id, metadata)

    @app.get("/iiif/3/{slide_id}/info.json", name="iiif_info")
    async def iiif_info(slide_id: str, request: Request) -> JSONResponse:
        _, metadata = await workers.run(metadata_for, slide_id)
        width, height = metadata.dimensions
        info_url = str(request.url_for("iiif_info", slide_id=slide_id))
        service_id = info_url.removesuffix("/info.json")
        return JSONResponse(
            {
                "@context": "http://iiif.io/api/image/3/context.json",
                "id": service_id,
                "type": "ImageService3",
                "protocol": "http://iiif.io/api/image",
                "profile": "level0",
                "width": width,
                "height": height,
                "tiles": [
                    {
                        "type": "Tile",
                        "width": tile_size,
                        "height": tile_size,
                        "scaleFactors": list(
                            iiif_scale_factors(metadata.dimensions, tile_size)
                        ),
                    }
                ],
                "preferredFormats": ["jpg"],
                "extraFormats": ["jpg", "png"],
                "extraQualities": ["default"],
                "extraFeatures": ["regionByPx", "sizeByW", "sizeByWh"],
            },
            headers={"Cache-Control": cache_control},
        )

    @app.get(
        "/iiif/3/{slide_id}/{region}/{size}/{rotation}/{quality}.{image_format}",
        name="iiif_image",
    )
    async def iiif_image(
        slide_id: str,
        region: str,
        size: str,
        rotation: str,
        quality: str,
        image_format: str,
        request: Request,
    ) -> Response:
        if rotation != "0" or quality != "default":
            raise HTTPException(
                status_code=400,
                detail="only rotation 0 and default quality are supported",
            )
        normalized_format = "jpg" if image_format == "jpeg" else image_format
        if normalized_format not in ("jpg", "png"):
            raise HTTPException(status_code=400, detail="format must be jpg or png")
        source, metadata = await workers.run(metadata_for, slide_id)
        try:
            parsed_region = _parse_region(region, metadata.dimensions)
            output_size = _parse_size(size, parsed_region)
            if not all(math.isfinite(value) for value in output_size):
                raise ValueError("invalid output size")
            encoded = await workers.run(
                renderer.render_region,
                source.path,
                parsed_region,
                output_size,
                image_format=normalized_format,
                source_mpp=source.source_mpp,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        headers = {"Cache-Control": cache_control, "ETag": encoded.etag}
        if request.headers.get("if-none-match") == encoded.etag:
            return Response(status_code=304, headers=headers)
        return Response(encoded.content, media_type=encoded.media_type, headers=headers)

    return app
