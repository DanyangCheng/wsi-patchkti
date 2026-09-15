"""Backend-neutral whole-slide image patch reading and sampling."""

from importlib.metadata import PackageNotFoundError, version

from .geometry import (
    ReadPlan,
    choose_level,
    plan_aligned_read,
    read_aligned_patch,
    virtual_canvas_size,
)
from .io import AutoSlideReader, OpenSlideReader, SlideReader, TiffReader
from .sampling import (
    GridSampler,
    IndexedSampler,
    PatchSampler,
    RandomSampler,
    TissueFilter,
    TissueMask,
    axis_positions,
)
from .stream import PatchStream
from .tiles import (
    EncodedImage,
    TileRenderer,
    choose_level_for_downsample,
    iiif_scale_factors,
)
from .types import (
    MPP,
    LevelInfo,
    Patch,
    PatchRequest,
    SamplingContext,
    Size,
    SlideMetadata,
    SlideSpec,
    as_mpp,
    as_size,
)

try:
    __version__ = version("wsi-patchkit")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "MPP",
    "Size",
    "AutoSlideReader",
    "EncodedImage",
    "GridSampler",
    "IndexedSampler",
    "LevelInfo",
    "OpenSlideReader",
    "Patch",
    "PatchRequest",
    "PatchSampler",
    "PatchStream",
    "RandomSampler",
    "ReadPlan",
    "SamplingContext",
    "SlideMetadata",
    "SlideReader",
    "SlideSpec",
    "TiffReader",
    "TileRenderer",
    "TissueFilter",
    "TissueMask",
    "as_mpp",
    "as_size",
    "axis_positions",
    "choose_level",
    "choose_level_for_downsample",
    "iiif_scale_factors",
    "plan_aligned_read",
    "read_aligned_patch",
    "virtual_canvas_size",
]
