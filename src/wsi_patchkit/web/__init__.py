"""Optional browser-based whole-slide image viewer."""

from .app import create_app
from .registry import SlideRegistry, SlideSource
from .workers import TileWorkerPool

__all__ = ["SlideRegistry", "SlideSource", "TileWorkerPool", "create_app"]
