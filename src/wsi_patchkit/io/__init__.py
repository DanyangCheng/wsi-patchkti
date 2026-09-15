"""WSI reader backends."""

from .base import SlideReader
from .openslide import OpenSlideReader
from .tiff import TiffReader

__all__ = ["OpenSlideReader", "SlideReader", "TiffReader"]
