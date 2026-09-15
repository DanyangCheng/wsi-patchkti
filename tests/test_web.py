from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx2")

from wsi_patchkit import TiffReader  # noqa: E402
from wsi_patchkit.web import SlideRegistry, TileWorkerPool, create_app  # noqa: E402


def _write_slide(path: Path) -> None:
    image = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(16, 16),
        resolution=(40_000, 40_000),
        resolutionunit="CENTIMETER",
        metadata=None,
    )


def test_registry_rejects_unsafe_ids(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    _write_slide(path)

    with pytest.raises(ValueError, match="slide IDs"):
        SlideRegistry({"../private": path})


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_viewer_serves_metadata_tiles_and_frontend(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    _write_slide(path)
    app = create_app({"case-001": path}, reader=TiffReader(), tile_size=8)

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        slides = await client.get("/api/slides")
        info = await client.get("/iiif/3/case-001/info.json")
        tile = await client.get(
            "/iiif/3/case-001/0,0,8,8/8,8/0/default.png"
        )
        cached = await client.get(
            "/iiif/3/case-001/0,0,8,8/8,8/0/default.png",
            headers={"If-None-Match": tile.headers["etag"]},
        )
        index = await client.get("/")
        script = await client.get("/static/app.js")

    assert slides.status_code == 200
    assert slides.json()[0]["id"] == "case-001"
    assert slides.json()[0]["mpp"] == [0.25, 0.25]
    assert info.status_code == 200
    assert info.json()["width"] == 20
    assert info.json()["tiles"][0]["scaleFactors"] == [1, 2, 4]
    assert tile.status_code == 200
    assert tile.headers["content-type"] == "image/png"
    assert tile.content.startswith(b"\x89PNG")
    assert cached.status_code == 304
    assert index.status_code == 200
    assert "WSI PatchKit Viewer" in index.text
    assert script.status_code == 200
    assert "dragToPan" in script.text


@pytest.mark.anyio
async def test_tile_worker_pool_runs_blocking_jobs_concurrently() -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0

    def work(value: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            return value * 2
        finally:
            with lock:
                active -= 1

    workers = TileWorkerPool(3)
    try:
        results = await asyncio.gather(
            *(workers.run(work, value) for value in range(3))
        )
    finally:
        workers.close()

    assert results == [0, 2, 4]
    assert max_active == 3
