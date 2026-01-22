"""
Configuration loader for the exoplanet detection pipeline.

This module provides utilities for loading and validating YAML configuration files.
All data fetching operations are driven by configuration files to ensure
reproducibility and flexibility.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RetryConfig:
    """Configuration for retry logic in network operations."""
    max_attempts: int = 3
    backoff_factor: float = 2.0
    timeout_seconds: int = 300


@dataclass
class GlobalConfig:
    """Global configuration settings."""
    data_root: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    metadata_dir: str = "data/metadata"
    logs_dir: str = "outputs/logs"
    retry: RetryConfig = field(default_factory=RetryConfig)
    show_progress: bool = True
    log_level: str = "INFO"


@dataclass
class TargetConfig:
    """Configuration for target selection."""
    kic_ids: list = field(default_factory=list)
    tic_ids: list = field(default_factory=list)
    sky_region: Optional[Dict[str, float]] = None
    star_names: list = field(default_factory=list)
    coordinates: list = field(default_factory=list)


@dataclass
class PhotometryConfig:
    """Configuration for space-based photometry data sources."""
    enabled: bool = True
    targets: TargetConfig = field(default_factory=TargetConfig)
    data_products: list = field(default_factory=lambda: ["SAP", "PDCSAP"])
    cadence: str = "auto"
    quality_filtering: bool = True
    quality_flags: list = field(default_factory=list)
    sectors: list = field(default_factory=list)
    output: Dict[str, str] = field(default_factory=dict)


@dataclass
class KeplerConfig(PhotometryConfig):
    """Kepler-specific configuration."""
    pass


@dataclass
class TESSConfig(PhotometryConfig):
    """TESS-specific configuration."""
    pass


@dataclass
class ExoplanetArchiveConfig:
    """Configuration for NASA Exoplanet Archive."""
    enabled: bool = True
    api_base_url: str = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    tables: list = field(default_factory=list)
    output: Dict[str, str] = field(default_factory=dict)


@dataclass
class RadialVelocityConfig:
    """Configuration for radial velocity data."""
    enabled: bool = False
    instruments: list = field(default_factory=list)
    targets: TargetConfig = field(default_factory=TargetConfig)
    output: Dict[str, str] = field(default_factory=dict)


@dataclass
class GaiaConfig:
    """Configuration for Gaia stellar data."""
    enabled: bool = True
    crossmatch_radius_arcsec: float = 5.0
    parameters: list = field(default_factory=list)
    output: Dict[str, str] = field(default_factory=dict)


@dataclass
class StellarContextConfig:
    """Configuration for stellar context data."""
    enabled: bool = False
    gaia: GaiaConfig = field(default_factory=GaiaConfig)


@dataclass
class DataConfig:
    """Complete data configuration structure."""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    photometry: Dict[str, Any] = field(default_factory=dict)
    exoplanet_archive: ExoplanetArchiveConfig = field(default_factory=ExoplanetArchiveConfig)
    radial_velocity: RadialVelocityConfig = field(default_factory=RadialVelocityConfig)
    stellar_context: StellarContextConfig = field(default_factory=StellarContextConfig)


def load_config(config_path: str) -> DataConfig:
    """
    Load and validate configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    
    Returns
    -------
    DataConfig
        Validated configuration object.
    
    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the configuration is invalid.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        raise ValueError("Configuration file is empty or invalid")
    
    # Extract global config
    global_dict = config_dict.get('global', {})
    retry_dict = global_dict.get('retry', {})
    retry_config = RetryConfig(
        max_attempts=retry_dict.get('max_attempts', 3),
        backoff_factor=retry_dict.get('backoff_factor', 2.0),
        timeout_seconds=retry_dict.get('timeout_seconds', 300)
    )
    global_config = GlobalConfig(
        data_root=global_dict.get('data_root', 'data'),
        raw_dir=global_dict.get('raw_dir', 'data/raw'),
        processed_dir=global_dict.get('processed_dir', 'data/processed'),
        metadata_dir=global_dict.get('metadata_dir', 'data/metadata'),
        logs_dir=global_dict.get('logs_dir', 'outputs/logs'),
        retry=retry_config,
        show_progress=global_dict.get('show_progress', True),
        log_level=global_dict.get('log_level', 'INFO')
    )
    
    # Extract photometry configs
    photometry_dict = config_dict.get('photometry', {})
    kepler_dict = photometry_dict.get('kepler', {})
    tess_dict = photometry_dict.get('tess', {})
    
    kepler_targets = TargetConfig(
        kic_ids=kepler_dict.get('targets', {}).get('kic_ids', []),
        sky_region=kepler_dict.get('targets', {}).get('sky_region')
    )
    kepler_config = KeplerConfig(
        enabled=kepler_dict.get('enabled', True),
        targets=kepler_targets,
        data_products=kepler_dict.get('data_products', ['SAP', 'PDCSAP']),
        cadence=kepler_dict.get('cadence', 'auto'),
        quality_filtering=kepler_dict.get('quality_filtering', True),
        quality_flags=kepler_dict.get('quality_flags', []),
        output=kepler_dict.get('output', {})
    )
    
    tess_targets = TargetConfig(
        tic_ids=tess_dict.get('targets', {}).get('tic_ids', []),
        sky_region=tess_dict.get('targets', {}).get('sky_region')
    )
    tess_config = TESSConfig(
        enabled=tess_dict.get('enabled', True),
        targets=tess_targets,
        data_products=tess_dict.get('data_products', ['SAP', 'PDCSAP']),
        cadence=tess_dict.get('cadence', 'auto'),
        quality_filtering=tess_dict.get('quality_filtering', True),
        sectors=tess_dict.get('sectors', []),
        output=tess_dict.get('output', {})
    )
    
    # Extract exoplanet archive config
    exo_dict = config_dict.get('exoplanet_archive', {})
    exoplanet_config = ExoplanetArchiveConfig(
        enabled=exo_dict.get('enabled', True),
        api_base_url=exo_dict.get('api_base_url', 
            'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI'),
        tables=exo_dict.get('tables', []),
        output=exo_dict.get('output', {})
    )
    
    # Extract radial velocity config
    rv_dict = config_dict.get('radial_velocity', {})
    rv_targets = TargetConfig(
        star_names=rv_dict.get('targets', {}).get('star_names', []),
        coordinates=rv_dict.get('targets', {}).get('coordinates', [])
    )
    rv_config = RadialVelocityConfig(
        enabled=rv_dict.get('enabled', False),
        instruments=rv_dict.get('instruments', []),
        targets=rv_targets,
        output=rv_dict.get('output', {})
    )
    
    # Extract stellar context config
    stellar_dict = config_dict.get('stellar_context', {})
    gaia_dict = stellar_dict.get('gaia', {})
    gaia_config = GaiaConfig(
        enabled=gaia_dict.get('enabled', True),
        crossmatch_radius_arcsec=gaia_dict.get('crossmatch_radius_arcsec', 5.0),
        parameters=gaia_dict.get('parameters', []),
        output=gaia_dict.get('output', {})
    )
    stellar_config = StellarContextConfig(
        enabled=stellar_dict.get('enabled', False),
        gaia=gaia_config
    )
    
    # Create complete config
    data_config = DataConfig(
        global_config=global_config,
        photometry={
            'kepler': kepler_config,
            'tess': tess_config
        },
        exoplanet_archive=exoplanet_config,
        radial_velocity=rv_config,
        stellar_context=stellar_config
    )
    
    # Validate configuration
    _validate_config(data_config)
    
    # Create necessary directories
    _create_directories(data_config)
    
    return data_config


def _validate_config(config: DataConfig) -> None:
    """
    Validate configuration for required fields and logical consistency.
    
    Parameters
    ----------
    config : DataConfig
        Configuration to validate.
    
    Raises
    ------
    ValueError
        If validation fails.
    """
    # Validate that at least one data source is enabled
    enabled_sources = []
    if config.photometry.get('kepler', {}).enabled:
        enabled_sources.append('kepler')
    if config.photometry.get('tess', {}).enabled:
        enabled_sources.append('tess')
    if config.exoplanet_archive.enabled:
        enabled_sources.append('exoplanet_archive')
    if config.radial_velocity.enabled:
        enabled_sources.append('radial_velocity')
    if config.stellar_context.enabled:
        enabled_sources.append('stellar_context')
    
    if not enabled_sources:
        raise ValueError("At least one data source must be enabled")
    
    # Validate photometry targets
    for name, phot_config in config.photometry.items():
        if phot_config.enabled:
            targets = phot_config.targets
            if name == 'kepler':
                has_targets = (targets.kic_ids or 
                              (targets.sky_region and 
                               targets.sky_region.get('ra') is not None))
            elif name == 'tess':
                has_targets = (targets.tic_ids or 
                              (targets.sky_region and 
                               targets.sky_region.get('ra') is not None))
            else:
                has_targets = True
            
            if not has_targets:
                raise ValueError(
                    f"{name} is enabled but no targets specified. "
                    "Provide either target IDs or sky region coordinates."
                )


def _create_directories(config: DataConfig) -> None:
    """
    Create necessary directory structure based on configuration.
    
    Parameters
    ----------
    config : DataConfig
        Configuration object.
    """
    dirs_to_create = [
        config.global_config.data_root,
        config.global_config.raw_dir,
        config.global_config.processed_dir,
        config.global_config.metadata_dir,
        config.global_config.logs_dir,
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
