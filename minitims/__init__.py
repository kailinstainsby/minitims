"""Mini-TIMs: Incremental rolling-window statistics for time-series."""

from .rolling import RollingWindow
from .stats import RollingMean, RollingVariance, RollingStdDev, RollingMin, RollingMax

__version__ = "0.1.0"
__all__ = ["RollingWindow", "RollingMean", "RollingVariance", "RollingStdDev", "RollingMin", "RollingMax"]
