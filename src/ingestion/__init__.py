"""
Data ingestion module for exoplanet detection pipeline.

This module provides downloaders for various astronomical data sources:
- Kepler and TESS photometry
- NASA Exoplanet Archive
- Radial velocity data
- Gaia stellar parameters
"""

from .base_loader import BaseLoader, DownloadMetadata
from .kepler_loader import KeplerLoader
from .tess_loader import TESSLoader
from .exoplanet_archive import ExoplanetArchiveLoader
from .radial_velocity import RadialVelocityLoader
from .gaia_loader import GaiaLoader

__all__ = [
    'BaseLoader',
    'DownloadMetadata',
    'KeplerLoader',
    'TESSLoader',
    'ExoplanetArchiveLoader',
    'RadialVelocityLoader',
    'GaiaLoader',
]

