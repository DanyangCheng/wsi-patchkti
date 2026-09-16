# wsi-patchkit

`wsi-patchkit` is a small, backend-neutral library for reading patches from
whole-slide images (WSIs). Version 0.1 provides:

- TIFF and optional OpenSlide readers with a common coordinate contract;
- physical-resolution (MPP) metadata and level selection;
- target-MPP-aligned, boundary-padded patch reads;
- deterministic grid, random, indexed, and tissue-filtered sampling;
- lazy patch streams;
- an optional PyTorch `IterableDataset` adapter;
- an optional browser viewer with IIIF tiles, zooming, and panning.

Dataset manifests, labels, augmentation, model execution, and prediction output
formats intentionally remain application concerns.

## Install

The project is managed with `uv`:

```bash
uv sync
```

Install the optional OpenSlide adapter when the native OpenSlide library is
already available:

```bash
uv sync --extra openslide
```

For platforms supported by the `openslide-bin` wheel, the convenience extra is:

```bash
uv sync --extra openslide-binary
```

PyTorch integration is isolated in another extra:

```bash
uv sync --extra torch
```

## Browser WSI viewer

Install the optional web dependencies:

```bash
uv sync --extra web
```

Start the viewer by explicitly registering one or more public slide IDs:

```bash
uv run wsi-patchkit-viewer \
  --slide case-001=/data/slides/case-001.svs \
  --slide case-002=/data/slides/case-002.tif \
  --reader-pool-size 4
```

To browse every supported WSI in a folder (including subfolders), register the
directory instead. The viewer's searchable slide menu scrolls when the list is
long:

```bash
uv run wsi-patchkit-viewer --slide-dir /data/slides
```

`--slide-dir` is repeatable and can be combined with `--slide`. Recognized file
extensions are `.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`, `.scn`, `.vms`,
`.vmu`, `.bif`, and `.qptiff`.

Then open <http://127.0.0.1:8000>. The viewer supports mouse-wheel and pinch
zooming, drag panning, double-click zooming, a navigator, level-0 coordinates,
and an MPP-aware scale bar. TIFF files use the bundled tifffile reader; other
formats are routed to the optional OpenSlide reader.

Applications can embed the viewer server instead of using the CLI:

```python
from wsi_patchkit.web import SlideSource, create_app

app = create_app(
    {
        "case-001": SlideSource("/data/slides/case-001.svs"),
        # Supply an override when the file has no reliable MPP metadata.
        "case-002": SlideSource("/data/slides/case-002.tif", source_mpp=0.5),
    }
)
```

The server exposes a conservative subset of IIIF Image API 3 at
`/iiif/3/{slide_id}`. Paths are registered server-side and are never accepted
from request URLs. Tiles are rendered from the closest suitable native pyramid
level and returned with an ETag and private cache headers. By default, four
independent readers process different tiles concurrently; tune
`--reader-pool-size` to match available CPU, memory, and storage throughput.

## Coordinate contract

Sampler coordinates live on a virtual canvas at `target_mpp`. For example, an
`x` coordinate of 100 at 0.5 MPP means 50 micrometres from the level-0 origin.
Reader-level `read_region()` calls use OpenSlide-compatible semantics: the
location is in level-0 pixels and the requested size is in pixels of the chosen
pyramid level. `read_aligned_patch()` performs the conversion explicitly.

## Sliding-window inference

```python
from wsi_patchkit import (
    GridSampler,
    PatchStream,
    SlideSpec,
    TiffReader,
)

reader = TiffReader()
metadata = reader.metadata("slide.tif", source_mpp=0.25)
slide = SlideSpec.from_metadata(metadata, target_mpp=0.5)
requests = GridSampler(patch_size=512, stride=256).sample([slide])

try:
    for patch in PatchStream(reader, requests):
        # patch.image is uint8 HWC; patch.request contains placement metadata.
        prediction = model(patch.image)
finally:
    reader.close()
```

The grid sampler aligns its final window with the far edge, so an inference
canvas is fully covered. Use `edge="drop"` if incomplete boundary regions should
be omitted instead.

## Reproducible random training samples

```python
from wsi_patchkit import RandomSampler, SamplingContext, SlideSpec

slides = [
    SlideSpec("a.svs", canvas_size=(120_000, 80_000), target_mpp=(0.5, 0.5)),
    SlideSpec("b.svs", canvas_size=(90_000, 70_000), target_mpp=(0.5, 0.5)),
]
sampler = RandomSampler(num_samples=20_000, patch_size=512, seed=2026)
requests = sampler.sample(slides, context=SamplingContext(epoch=3))
```

Random samples are derived from `(seed, epoch, global sample index)`. Changing
DataLoader worker count does not change the global sample set; each worker only
receives its deterministic shard.

## Tissue filtering

```python
from wsi_patchkit import TissueFilter, TissueMask

masks = {
    "a.svs": TissueMask(mask_array, canvas_size=(120_000, 80_000)),
}
filtered = TissueFilter(masks, minimum_fraction=0.05).filter(requests)
```

The tissue mask may be low resolution; it is mapped over the corresponding
target-MPP canvas. Version 0.1 consumes caller-provided masks and does not impose
a tissue-detection algorithm.

## Indexed sampling

`IndexedSampler` samples precomputed `PatchRequest` objects. Index serialization
is deliberately left to the caller in v0.1 so projects can use NPZ, Parquet, or
a database without coupling the core library to one storage format.

## PyTorch adapter

```python
from wsi_patchkit import RandomSampler, SlideSpec
from wsi_patchkit.io import TiffReader
from wsi_patchkit.torch import WSIPatchIterableDataset

dataset = WSIPatchIterableDataset(
    slides,
    RandomSampler(num_samples=20_000, patch_size=512, seed=2026),
    reader_factory=TiffReader,
)
```

Each worker creates and closes its own reader. Items contain a float32 CHW
`image` tensor in `[0, 1]` plus the original `PatchRequest`.

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest --cov
uv build
```
