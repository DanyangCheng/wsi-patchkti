"""WSI reader backends."""

from .auto import AutoSlideReader
from .base import SlideReader
from .openslide import OpenSlideReader
from .tiff import TiffReader

__all__ = ["AutoSlideReader", "OpenSlideReader", "SlideReader", "TiffReader"]
