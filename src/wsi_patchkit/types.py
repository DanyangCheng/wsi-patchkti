"""Public value objects shared by readers, samplers, and streams."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

MPP = tuple[float, float]
Size = tuple[int, int]


def as_mpp(value: float | Sequence[float], *, name: str = "mpp") -> MPP:
    """Validate a scalar or X/Y pair and return an MPP pair."""
    if isinstance(value, (int, float)):
        pair = float(value), float(value)
    else:
        pair = tuple(float(item) for item in value)
        if len(pair) != 2:
            raise ValueError(f"{name} must be a scalar or an X/Y pair")
    if not all(math.isfinite(item) and item > 0 for item in pair):
        raise ValueError(f"{name} must contain finite positive values, got {value!r}")
    return pair


def as_size(value: int | Sequence[int], *, name: str = "size") -> Size:
    """Validate a scalar or width/height pair."""
    if isinstance(value, int):
        pair = value, value
    else:
        pair = tuple(int(item) for item in value)
        if len(pair) != 2:
            raise ValueError(f"{name} must be an integer or a width/height pair")
    if any(item < 1 for item in pair):
        raise ValueError(f"{name} must contain positive integers, got {value!r}")
    return pair


@dataclass(frozen=True, slots=True)
class LevelInfo:
    """Geometry and optional physical resolution for one pyramid level."""

    level: int
    dimensions: Size
    downsample: tuple[float, float]
    mpp: MPP | None = None

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("level must be non-negative")
        object.__setattr__(
            self, "dimensions", as_size(self.dimensions, name="dimensions")
        )
        downsample = tuple(float(item) for item in self.downsample)
        if len(downsample) != 2 or not all(
            math.isfinite(item) and item > 0 for item in downsample
        ):
            raise ValueError("downsample must contain two finite positive values")
        object.__setattr__(self, "downsample", downsample)
        if self.mpp is not None:
            object.__setattr__(self, "mpp", as_mpp(self.mpp))


@dataclass(frozen=True, slots=True)
class SlideMetadata:
    """Backend-neutral WSI metadata."""

    path: str | Path
    levels: tuple[LevelInfo, ...]
    mpp: MPP | None = None
    vendor: str | None = None
    properties: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = str(Path(self.path).resolve())
        if not self.levels:
            raise ValueError("slide metadata must contain at least one level")
        if tuple(level.level for level in self.levels) != tuple(
            range(len(self.levels))
        ):
            raise ValueError("levels must be ordered and numbered from zero")
        object.__setattr__(self, "path", path)
        if self.mpp is not None:
            object.__setattr__(self, "mpp", as_mpp(self.mpp))
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))

    @property
    def dimensions(self) -> Size:
        return self.levels[0].dimensions


@dataclass(frozen=True, slots=True)
class SlideSpec:
    """A slide projected onto a virtual canvas at a requested MPP."""

    path: str | Path
    canvas_size: Size
    target_mpp: MPP | float
    source_mpp: MPP | float | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(Path(self.path)))
        object.__setattr__(
            self, "canvas_size", as_size(self.canvas_size, name="canvas_size")
        )
        object.__setattr__(
            self, "target_mpp", as_mpp(self.target_mpp, name="target_mpp")
        )
        if self.source_mpp is not None:
            object.__setattr__(
                self, "source_mpp", as_mpp(self.source_mpp, name="source_mpp")
            )
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and non-negative")
        object.__setattr__(self, "weight", weight)

    @classmethod
    def from_metadata(
        cls,
        metadata: SlideMetadata,
        *,
        target_mpp: float | Sequence[float],
        weight: float = 1.0,
    ) -> SlideSpec:
        """Construct a virtual-canvas specification from physical metadata."""
        if metadata.mpp is None:
            raise ValueError("slide has no MPP metadata; provide source_mpp explicitly")
        target = as_mpp(target_mpp, name="target_mpp")
        width, height = metadata.dimensions
        canvas = (
            max(1, round(width * metadata.mpp[0] / target[0])),
            max(1, round(height * metadata.mpp[1] / target[1])),
        )
        return cls(
            metadata.path,
            canvas_size=canvas,
            target_mpp=target,
            source_mpp=metadata.mpp,
            weight=weight,
        )


@dataclass(frozen=True, slots=True)
class PatchRequest:
    """One patch on a target-MPP virtual canvas."""

    slide: str | Path
    x: int
    y: int
    width: int
    height: int
    target_mpp: MPP | float
    source_mpp: MPP | float | None = None
    fill_value: int = 255

    def __post_init__(self) -> None:
        object.__setattr__(self, "slide", str(Path(self.slide)))
        for name in ("x", "y", "width", "height", "fill_value"):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.width < 1 or self.height < 1:
            raise ValueError("patch width and height must be positive")
        if not 0 <= self.fill_value <= 255:
            raise ValueError("fill_value must be in [0, 255]")
        object.__setattr__(
            self, "target_mpp", as_mpp(self.target_mpp, name="target_mpp")
        )
        if self.source_mpp is not None:
            object.__setattr__(
                self, "source_mpp", as_mpp(self.source_mpp, name="source_mpp")
            )


@dataclass(frozen=True, slots=True)
class Patch:
    """A materialized patch and the request that produced it."""

    request: PatchRequest
    image: object


@dataclass(frozen=True, slots=True)
class SamplingContext:
    """Distributed and worker context used for deterministic sharding."""

    epoch: int = 0
    rank: int = 0
    world_size: int = 1
    worker_id: int = 0
    num_workers: int = 1

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.world_size < 1 or self.num_workers < 1:
            raise ValueError("world_size and num_workers must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if not 0 <= self.worker_id < self.num_workers:
            raise ValueError("worker_id must be in [0, num_workers)")

    @property
    def shard_index(self) -> int:
        return self.rank * self.num_workers + self.worker_id

    @property
    def shard_count(self) -> int:
        return self.world_size * self.num_workers

    def owns(self, global_index: int) -> bool:
        return global_index % self.shard_count == self.shard_index
