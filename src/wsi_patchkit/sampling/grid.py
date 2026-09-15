"""Deterministic sliding-window sampling."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from ..types import PatchRequest, SamplingContext, SlideSpec, as_size


def axis_positions(
    length: int,
    patch_size: int,
    stride: int,
    *,
    edge: Literal["align", "drop"] = "align",
) -> tuple[int, ...]:
    """Generate positions for one axis."""
    if min(length, patch_size, stride) < 1:
        raise ValueError("length, patch_size, and stride must be positive")
    if edge == "drop":
        if length < patch_size:
            return ()
        return tuple(range(0, length - patch_size + 1, stride))
    if edge != "align":
        raise ValueError("edge must be 'align' or 'drop'")
    if length <= patch_size:
        return (0,)
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return tuple(positions)


@dataclass(frozen=True, slots=True)
class GridSampler:
    """Enumerate a complete, optionally edge-aligned patch grid."""

    patch_size: int | tuple[int, int]
    stride: int | tuple[int, int] | None = None
    edge: Literal["align", "drop"] = "align"
    fill_value: int = 255

    def __post_init__(self) -> None:
        patch_size = as_size(self.patch_size, name="patch_size")
        stride = (
            patch_size if self.stride is None else as_size(self.stride, name="stride")
        )
        if self.edge not in ("align", "drop"):
            raise ValueError("edge must be 'align' or 'drop'")
        if self.edge == "align" and any(
            step > patch for step, patch in zip(stride, patch_size, strict=True)
        ):
            raise ValueError("align stride must not exceed patch_size")
        if not 0 <= self.fill_value <= 255:
            raise ValueError("fill_value must be in [0, 255]")
        object.__setattr__(self, "patch_size", patch_size)
        object.__setattr__(self, "stride", stride)

    def sample(
        self,
        slides: Iterable[SlideSpec],
        *,
        context: SamplingContext | None = None,
    ) -> Iterator[PatchRequest]:
        context = context or SamplingContext()
        patch_width, patch_height = self.patch_size
        stride_x, stride_y = self.stride
        global_index = 0
        for slide in slides:
            width, height = slide.canvas_size
            for y in axis_positions(
                height,
                patch_height,
                stride_y,
                edge=self.edge,
            ):
                for x in axis_positions(
                    width,
                    patch_width,
                    stride_x,
                    edge=self.edge,
                ):
                    if context.owns(global_index):
                        yield PatchRequest(
                            slide.path,
                            x,
                            y,
                            patch_width,
                            patch_height,
                            slide.target_mpp,
                            source_mpp=slide.source_mpp,
                            fill_value=self.fill_value,
                        )
                    global_index += 1
