"""Tracking pipeline package."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '.*' or newer of 'numexpr'.*",
    category=UserWarning,
    module=r"pandas\.core\.computation\.expressions",
)
warnings.filterwarnings(
    "ignore",
    message=r"Pandas requires version '.*' or newer of 'bottleneck'.*",
    category=UserWarning,
    module=r"pandas\.core\.arrays\.masked",
)

__all__ = ["__version__"]

__version__ = "0.1.0"
