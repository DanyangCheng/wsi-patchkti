"""Backend-neutral whole-slide image patch reading and sampling."""

from importlib.metadata import PackageNotFoundError, version

from .geometry import (
    Interpolation,
    LevelSelectionPolicy,
    PatchReadResult,
    ReadPlan,
    choose_level,
    plan_aligned_read,
    read_aligned_patch,
    read_aligned_patch_result,
    virtual_canvas_size,
)
from .io import AutoSlideReader, OpenSlideReader, SlideReader, TiffReader
from .sampling import (
    GridSampler,
    IndexedSampler,
    PatchRequestSource,
    PatchSampler,
    RandomSampler,
    TissueFilter,
    TissueMask,
    TissueRandomSampler,
    axis_positions,
)
from .stream import PatchStream
from .tiles import (
    EncodedImage,
    SlideReaderPool,
    TileRenderer,
    choose_level_for_downsample,
    iiif_scale_factors,
)
from .types import (
    MPP,
    LevelInfo,
    Patch,
    PatchRequest,
    PixelFormat,
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
    "Interpolation",
    "LevelSelectionPolicy",
    "LevelInfo",
    "OpenSlideReader",
    "Patch",
    "PatchReadResult",
    "PatchRequest",
    "PatchRequestSource",
    "PatchSampler",
    "PatchStream",
    "PixelFormat",
    "RandomSampler",
    "ReadPlan",
    "SamplingContext",
    "SlideMetadata",
    "SlideReader",
    "SlideReaderPool",
    "SlideSpec",
    "TiffReader",
    "TileRenderer",
    "TissueFilter",
    "TissueMask",
    "TissueRandomSampler",
    "as_mpp",
    "as_size",
    "axis_positions",
    "choose_level",
    "choose_level_for_downsample",
    "iiif_scale_factors",
    "plan_aligned_read",
    "read_aligned_patch",
    "read_aligned_patch_result",
    "virtual_canvas_size",
]
