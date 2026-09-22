"""Small, dependency-free benchmarks for sampling and optional TIFF patch I/O.

Run sampling-only measurements with ``uv run python benchmarks/benchmark_patchkit.py``.
Pass ``--slide /path/to/slide.tif`` to include aligned TIFF patch reads.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

from wsi_patchkit import (
    GridPatchRequestSampler,
    PatchStream,
    RandomPatchRequestSampler,
    SlideSpec,
    TiffReader,
)


def _seconds(callback: Callable[[], object]) -> float:
    start = time.perf_counter()
    callback()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--slide", type=Path)
    parser.add_argument("--source-mpp", type=float)
    arguments = parser.parse_args()
    if arguments.samples < 1 or arguments.patch_size < 1:
        parser.error("--samples and --patch-size must be positive")

    synthetic = SlideSpec(
        "benchmark.tif",
        canvas_size=(100_000, 80_000),
        target_mpp=0.5,
    )
    random_sampler = RandomPatchRequestSampler(
        arguments.samples, arguments.patch_size, seed=2026
    )
    grid_sampler = GridPatchRequestSampler(
        arguments.patch_size, stride=arguments.patch_size
    )
    random_seconds = _seconds(lambda: list(random_sampler.sample([synthetic])))
    grid_requests: list[object] = []

    def materialize_grid() -> None:
        nonlocal grid_requests
        grid_requests = list(grid_sampler.sample([synthetic]))

    grid_seconds = _seconds(materialize_grid)
    print(f"random: {arguments.samples / random_seconds:,.0f} requests/s")
    print(f"grid: {len(grid_requests) / grid_seconds:,.0f} requests/s")

    if arguments.slide is None:
        return
    reader = TiffReader()
    try:
        metadata = reader.metadata(arguments.slide, source_mpp=arguments.source_mpp)
        slide = SlideSpec.from_metadata(metadata, target_mpp=0.5)
        requests = list(
            RandomPatchRequestSampler(
                arguments.samples, arguments.patch_size, seed=2026
            ).sample(
                [slide]
            )
        )
        read_seconds = _seconds(lambda: list(PatchStream(reader, requests)))
        print(f"aligned TIFF reads: {len(requests) / read_seconds:,.1f} patches/s")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
