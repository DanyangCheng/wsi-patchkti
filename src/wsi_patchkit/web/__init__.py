"""Optional browser-based whole-slide image viewer."""

from .app import create_app
from .registry import SlideRegistry, SlideSource

__all__ = ["SlideRegistry", "SlideSource", "create_app"]

