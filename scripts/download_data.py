#!/usr/bin/env python3
"""
Main orchestration script for downloading all exoplanet detection data.

This script coordinates the download of data from multiple sources:
- Kepler and TESS photometry
- NASA Exoplanet Archive labels
- Radial velocity data (optional)
- Gaia stellar parameters (optional)

Usage:
    python scripts/download_data.py --config configs/data.yaml
    python scripts/download_data.py --config configs/data.yaml --skip-cache
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.ingestion.kepler_loader import KeplerLoader
from src.ingestion.tess_loader import TESSLoader
from src.ingestion.exoplanet_archive import ExoplanetArchiveLoader
from src.ingestion.radial_velocity import RadialVelocityLoader
from src.ingestion.gaia_loader import GaiaLoader


def setup_logging(config: Any) -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters
    ----------
    config : Any
        Global configuration object.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path(config.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    return logger


def download_photometry(config: Any, logger: logging.Logger, skip_cache: bool = False) -> Dict[str, Any]:
    """
    Download photometry data (Kepler and/or TESS).
    
    Parameters
    ----------
    config : Any
        Data configuration object.
    logger : logging.Logger
        Logger instance.
    skip_cache : bool
        Whether to skip cached files.
    
    Returns
    -------
    Dict[str, Any]
        Summary of downloads.
    """
    results = {}
    
    # Kepler
    if config.photometry.get('kepler', {}).enabled:
        logger.info("=" * 60)
        logger.info("Downloading Kepler data")
        logger.info("=" * 60)
        
        try:
            kepler_config = config.photometry['kepler']
            kepler_loader = KeplerLoader(kepler_config, config.global_config, logger)
            
            metadata = kepler_loader.download()
            results['kepler'] = {
                'success': len(metadata.errors) == 0,
                'num_files': metadata.num_files,
                'size_mb': metadata.data_size_mb,
                'errors': metadata.errors
            }
            
            logger.info(f"Kepler download complete: {metadata.num_files} files, {metadata.data_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Kepler download failed: {e}", exc_info=True)
            results['kepler'] = {'success': False, 'error': str(e)}
    
    # TESS
    if config.photometry.get('tess', {}).enabled:
        logger.info("=" * 60)
        logger.info("Downloading TESS data")
        logger.info("=" * 60)
        
        try:
            tess_config = config.photometry['tess']
            tess_loader = TESSLoader(tess_config, config.global_config, logger)
            
            metadata = tess_loader.download()
            results['tess'] = {
                'success': len(metadata.errors) == 0,
                'num_files': metadata.num_files,
                'size_mb': metadata.data_size_mb,
                'errors': metadata.errors
            }
            
            logger.info(f"TESS download complete: {metadata.num_files} files, {metadata.data_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"TESS download failed: {e}", exc_info=True)
            results['tess'] = {'success': False, 'error': str(e)}
    
    return results


def download_exoplanet_archive(config: Any, logger: logging.Logger, skip_cache: bool = False) -> Dict[str, Any]:
    """
    Download exoplanet archive data.
    
    Parameters
    ----------
    config : Any
        Data configuration object.
    logger : logging.Logger
        Logger instance.
    skip_cache : bool
        Whether to skip cached files.
    
    Returns
    -------
    Dict[str, Any]
        Summary of downloads.
    """
    results = {}
    
    if config.exoplanet_archive.enabled:
        logger.info("=" * 60)
        logger.info("Downloading NASA Exoplanet Archive data")
        logger.info("=" * 60)
        
        try:
            exo_loader = ExoplanetArchiveLoader(
                config.exoplanet_archive,
                config.global_config,
                logger
            )
            
            metadata = exo_loader.download()
            results['exoplanet_archive'] = {
                'success': len(metadata.errors) == 0,
                'num_files': metadata.num_files,
                'size_mb': metadata.data_size_mb,
                'errors': metadata.errors
            }
            
            logger.info(
                f"Exoplanet Archive download complete: "
                f"{metadata.num_files} files, {metadata.data_size_mb:.2f} MB"
            )
            
        except Exception as e:
            logger.error(f"Exoplanet Archive download failed: {e}", exc_info=True)
            results['exoplanet_archive'] = {'success': False, 'error': str(e)}
    
    return results


def download_radial_velocity(config: Any, logger: logging.Logger, skip_cache: bool = False) -> Dict[str, Any]:
    """
    Download radial velocity data.
    
    Parameters
    ----------
    config : Any
        Data configuration object.
    logger : logging.Logger
        Logger instance.
    skip_cache : bool
        Whether to skip cached files.
    
    Returns
    -------
    Dict[str, Any]
        Summary of downloads.
    """
    results = {}
    
    if config.radial_velocity.enabled:
        logger.info("=" * 60)
        logger.info("Downloading Radial Velocity data")
        logger.info("=" * 60)
        
        try:
            rv_loader = RadialVelocityLoader(
                config.radial_velocity,
                config.global_config,
                logger
            )
            
            metadata = rv_loader.download()
            results['radial_velocity'] = {
                'success': len(metadata.errors) == 0,
                'num_files': metadata.num_files,
                'size_mb': metadata.data_size_mb,
                'errors': metadata.errors
            }
            
            logger.info(
                f"Radial Velocity download complete: "
                f"{metadata.num_files} files, {metadata.data_size_mb:.2f} MB"
            )
            
        except Exception as e:
            logger.error(f"Radial Velocity download failed: {e}", exc_info=True)
            results['radial_velocity'] = {'success': False, 'error': str(e)}
    
    return results


def download_gaia(config: Any, logger: logging.Logger, skip_cache: bool = False) -> Dict[str, Any]:
    """
    Download Gaia stellar data.
    
    Parameters
    ----------
    config : Any
        Data configuration object.
    logger : logging.Logger
        Logger instance.
    skip_cache : bool
        Whether to skip cached files.
    
    Returns
    -------
    Dict[str, Any]
        Summary of downloads.
    """
    results = {}
    
    if config.stellar_context.enabled and config.stellar_context.gaia.enabled:
        logger.info("=" * 60)
        logger.info("Downloading Gaia stellar data")
        logger.info("=" * 60)
        
        try:
            # Note: Gaia requires coordinates, which should come from photometry data
            # For now, this is a placeholder that would need to be extended
            # to read coordinates from downloaded photometry metadata
            
            gaia_loader = GaiaLoader(
                config.stellar_context.gaia,
                config.global_config,
                logger
            )
            
            # This would need coordinates from photometry data
            # For demonstration, we'll skip if no coordinates provided
            logger.warning(
                "Gaia download requires coordinates. "
                "This should be integrated with photometry data download."
            )
            
            results['gaia'] = {
                'success': False,
                'error': 'Gaia download requires coordinates from photometry data'
            }
            
        except Exception as e:
            logger.error(f"Gaia download failed: {e}", exc_info=True)
            results['gaia'] = {'success': False, 'error': str(e)}
    
    return results


def main():
    """Main entry point for data download script."""
    parser = argparse.ArgumentParser(
        description="Download exoplanet detection data from multiple sources"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration YAML file'
    )
    parser.add_argument(
        '--skip-cache',
        action='store_true',
        help='Skip cached files and re-download'
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['kepler', 'tess', 'exoplanet_archive', 'radial_velocity', 'gaia', 'all'],
        default=['all'],
        help='Which data sources to download (default: all)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set up logging
    logger = setup_logging(config.global_config)
    
    # Set random seeds for reproducibility
    from src.utils.seeding import set_all_seeds
    seed = getattr(config.global_config, 'seed', 42)
    set_all_seeds(seed)
    logger.info(f"Global seed set to {seed}")
    
    logger.info("=" * 60)
    logger.info("Exoplanet Detection Data Download")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Sources: {args.sources}")
    logger.info(f"Skip cache: {args.skip_cache}")
    
    # Determine which sources to download
    download_all = 'all' in args.sources
    sources_to_download = set(args.sources) if not download_all else set(['all'])
    
    # Collect results
    all_results = {}
    
    # Download photometry
    if download_all or any(s in sources_to_download for s in ['kepler', 'tess', 'all']):
        photometry_results = download_photometry(config, logger, args.skip_cache)
        all_results.update(photometry_results)
    
    # Download exoplanet archive
    if download_all or 'exoplanet_archive' in sources_to_download:
        exo_results = download_exoplanet_archive(config, logger, args.skip_cache)
        all_results.update(exo_results)
    
    # Download radial velocity
    if download_all or 'radial_velocity' in sources_to_download:
        rv_results = download_radial_velocity(config, logger, args.skip_cache)
        all_results.update(rv_results)
    
    # Download Gaia
    if download_all or 'gaia' in sources_to_download:
        gaia_results = download_gaia(config, logger, args.skip_cache)
        all_results.update(gaia_results)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    
    total_files = 0
    total_size_mb = 0.0
    total_errors = 0
    
    for source, result in all_results.items():
        if result.get('success', False):
            logger.info(f"{source}: SUCCESS - {result.get('num_files', 0)} files, {result.get('size_mb', 0):.2f} MB")
            total_files += result.get('num_files', 0)
            total_size_mb += result.get('size_mb', 0.0)
        else:
            logger.error(f"{source}: FAILED - {result.get('error', 'Unknown error')}")
            total_errors += 1
        
        if result.get('errors'):
            for error in result['errors']:
                logger.warning(f"  - {error}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {total_files} files, {total_size_mb:.2f} MB, {total_errors} failures")
    
    # Save summary to metadata directory
    summary_path = Path(config.global_config.metadata_dir) / "download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config_file': args.config,
            'results': all_results,
            'summary': {
                'total_files': total_files,
                'total_size_mb': total_size_mb,
                'total_errors': total_errors
            }
        }, f, indent=2, default=str)
    
    logger.info(f"Summary saved to {summary_path}")
    
    # Save config snapshot for traceability
    import shutil
    metadata_dir = Path(config.global_config.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    config_copy_path = metadata_dir / f"config_{timestamp}.yaml"
    shutil.copy(args.config, config_copy_path)
    logger.info(f"Saved config snapshot to {config_copy_path}")
    
    # Exit with error code if any failures
    if total_errors > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
