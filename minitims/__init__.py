"""Mini-TIMs: Incremental rolling-window statistics for time-series."""

from .rolling import RollingWindow
from .stats import RollingMean

__version__ = "0.1.0"
__all__ = ["RollingWindow", "RollingMean"]
