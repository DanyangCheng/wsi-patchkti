from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx2")

from wsi_patchkit import TiffReader  # noqa: E402
from wsi_patchkit.web import SlideRegistry, TileWorkerPool, create_app  # noqa: E402
from wsi_patchkit.web.__main__ import discover_slides  # noqa: E402


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


def test_discover_slides_recurses_filters_and_creates_unique_ids(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested folder"
    nested.mkdir()
    first = tmp_path / "case 1.SVS"
    second = nested / "case 1.SVS"
    ignored = tmp_path / "notes.txt"
    first.touch()
    second.touch()
    ignored.touch()

    slides = discover_slides([tmp_path])

    assert list(slides) == ["case_1.SVS", "nested_folder_case_1.SVS"]
    assert set(slides.values()) == {first, second}


def test_discover_slides_disambiguates_ids_across_directories(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "case.svs").touch()
    (second_dir / "case.svs").touch()

    slides = discover_slides([first_dir, second_dir])

    assert list(slides) == ["case.svs", "case.svs-2"]


@pytest.mark.anyio
async def test_slide_list_does_not_open_slides(tmp_path: Path) -> None:
    path = tmp_path / "unreadable.svs"
    path.touch()
    app = create_app({"unreadable.svs": path}, reader=TiffReader())

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/slides")

    assert response.status_code == 200
    assert response.json() == [{"id": "unreadable.svs"}]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _wait_for_crop(client, slide_id: str, job_id: str) -> dict[str, object]:
    for _ in range(200):
        response = await client.get(f"/api/slides/{slide_id}/crops/{job_id}")
        assert response.status_code == 200
        result = response.json()
        if result["status"] in ("completed", "failed"):
            return result
        await asyncio.sleep(0.01)
    pytest.fail("crop job did not finish")


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
        metadata = await client.get("/api/slides/case-001")
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
    assert slides.json() == [{"id": "case-001"}]
    assert metadata.status_code == 200
    assert metadata.json()["mpp"] == [0.25, 0.25]
    assert info.status_code == 200
    assert info.json()["width"] == 20
    assert info.json()["tiles"][0]["scaleFactors"] == [1, 2, 4]
    assert tile.status_code == 200
    assert tile.headers["content-type"] == "image/png"
    assert tile.content.startswith(b"\x89PNG")
    assert cached.status_code == 304
    assert index.status_code == 200
    assert "WSI PatchKit Viewer" in index.text
    assert "/static/app.js?v=5" in index.text
    assert script.status_code == 200
    assert "dragToPan" in script.text
    assert "populateSlideMenu" in script.text
    assert "saveCrop" in script.text
    assert "pollCropJob" in script.text
    assert "wsi-patchkit.crop-size" in script.text


@pytest.mark.anyio
async def test_level0_crop_is_saved_on_server(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    output_dir = tmp_path / "saved-crops"
    _write_slide(path)
    app = create_app(
        {"case-001": path},
        reader=TiffReader(),
        crop_output_dir=output_dir,
    )

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/slides/case-001/crops",
            json={
                "x": 3,
                "y": 4,
                "width": 6,
                "height": 5,
                "format": "png",
                "filename": "manual-crop.png",
            },
        )
        assert response.status_code == 202
        submitted = response.json()
        completed = await _wait_for_crop(client, "case-001", submitted["job_id"])

    assert submitted == {
        "job_id": submitted["job_id"],
        "status": "queued",
        "slide_id": "case-001",
        "filename": "manual-crop.png",
        "format": "png",
        "level": 0,
        "region": {"x": 3, "y": 4, "width": 6, "height": 5},
    }
    assert completed["status"] == "completed"
    destination = output_dir / "manual-crop.png"
    assert destination.is_file()
    expected = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    np.testing.assert_array_equal(
        np.asarray(Image.open(destination)),
        expected[4:9, 3:9],
    )


@pytest.mark.anyio
async def test_crop_requests_queue_without_waiting_for_previous_save(
    tmp_path: Path,
) -> None:
    path = tmp_path / "slide.tif"
    output_dir = tmp_path / "saved-crops"
    _write_slide(path)
    started = threading.Event()
    release = threading.Event()

    class BlockingReader:
        def __init__(self) -> None:
            self.delegate = TiffReader()

        def metadata(self, value, *, source_mpp=None):
            return self.delegate.metadata(value, source_mpp=source_mpp)

        def read_region(self, value, location, level, size):
            started.set()
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release crop reader")
            return self.delegate.read_region(value, location, level, size)

        def close(self) -> None:
            self.delegate.close()

    app = create_app(
        {"case-001": path},
        reader=BlockingReader(),
        crop_output_dir=output_dir,
        crop_workers=1,
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            first = await client.post(
                "/api/slides/case-001/crops",
                json={
                    "x": 0,
                    "y": 0,
                    "width": 3,
                    "height": 3,
                    "filename": "first.png",
                },
            )
            assert first.status_code == 202
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()

            second = await client.post(
                "/api/slides/case-001/crops",
                json={
                    "x": 3,
                    "y": 3,
                    "width": 3,
                    "height": 3,
                    "filename": "second.png",
                },
            )
            assert second.status_code == 202
            first_status = await client.get(
                f"/api/slides/case-001/crops/{first.json()['job_id']}"
            )
            second_status = await client.get(
                f"/api/slides/case-001/crops/{second.json()['job_id']}"
            )
            assert first_status.json()["status"] == "running"
            assert second_status.json()["status"] == "queued"

            release.set()
            first_done = await _wait_for_crop(
                client,
                "case-001",
                first.json()["job_id"],
            )
            second_done = await _wait_for_crop(
                client,
                "case-001",
                second.json()["job_id"],
            )
            assert first_done["status"] == "completed"
            assert second_done["status"] == "completed"
    finally:
        release.set()

    assert {path.name for path in output_dir.iterdir()} == {"first.png", "second.png"}


@pytest.mark.anyio
async def test_crop_rejects_out_of_bounds_and_unsafe_filenames(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    output_dir = tmp_path / "saved-crops"
    _write_slide(path)
    app = create_app(
        {"case-001": path},
        reader=TiffReader(),
        crop_output_dir=output_dir,
    )

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        outside = await client.post(
            "/api/slides/case-001/crops",
            json={"x": 18, "y": 0, "width": 3, "height": 3},
        )
        unsafe = await client.post(
            "/api/slides/case-001/crops",
            json={
                "x": 0,
                "y": 0,
                "width": 3,
                "height": 3,
                "filename": "../escape.png",
            },
        )

    assert outside.status_code == 400
    assert "fully inside" in outside.json()["detail"]
    assert unsafe.status_code == 400
    assert not output_dir.exists()


@pytest.mark.anyio
async def test_backend_render_errors_are_not_reported_as_bad_requests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "slide.tif"
    _write_slide(path)

    class BrokenReader:
        def __init__(self) -> None:
            self.delegate = TiffReader()

        def metadata(self, value, *, source_mpp=None):
            return self.delegate.metadata(value, source_mpp=source_mpp)

        def read_region(self, value, location, level, size):
            raise ValueError("JPEG decoding requires imagecodecs")

        def close(self) -> None:
            self.delegate.close()

    app = create_app({"case-001": path}, reader=BrokenReader(), tile_size=8)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        backend_failure = await client.get(
            "/iiif/3/case-001/0,0,8,8/8,8/0/default.png"
        )
        invalid_region = await client.get(
            "/iiif/3/case-001/20,0,8,8/8,8/0/default.png"
        )

    assert backend_failure.status_code == 422
    assert backend_failure.json()["detail"] == (
        "registered slide tile could not be rendered"
    )
    assert invalid_region.status_code == 400
    assert invalid_region.json()["detail"] == "region starts outside the image"


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
