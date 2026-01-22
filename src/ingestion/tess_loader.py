"""
TESS (Transiting Exoplanet Survey Satellite) data downloader.

This module provides functionality to download and process light curve data
from the TESS mission via the MAST archive.

Data Source:
    - Mission: TESS (Transiting Exoplanet Survey Satellite, NASA)
    - Archive: MAST (Mikulski Archive for Space Telescopes)
    - Access: Via lightkurve and astroquery packages

Astrophysical Context:
    TESS observes the entire sky in 27-day sectors, focusing on bright stars
    near the ecliptic. It has discovered thousands of exoplanet candidates.
    Like Kepler, TESS provides:
    - SAP (Simple Aperture Photometry): Raw flux measurements
    - PDCSAP (Pre-search Data Conditioning SAP): Systematics-corrected flux

Assumptions:
    - Targets are identified by TIC (TESS Input Catalog) IDs
    - Data is organized by sectors (27-day observing periods)
    - Multiple cadences available (short, fast)
    - Quality flags follow TESS standard definitions

Limitations:
    - Sector-based coverage means targets may have gaps
    - Some targets observed in multiple sectors
    - Large datasets require significant storage
"""

import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

try:
    import lightkurve as lk
    from lightkurve import LightCurve
except ImportError:
    raise ImportError(
        "lightkurve is required for TESS data download. "
        "Install with: pip install lightkurve"
    )

from .base_loader import BaseLoader, DownloadMetadata


class TESSLoader(BaseLoader):
    """
    Downloader for TESS light curve data.
    
    This class handles downloading, parsing, and storing TESS light curves
    from the MAST archive using the lightkurve package.
    """
    
    def __init__(self, config: Any, global_config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize TESS loader.
        
        Parameters
        ----------
        config : Any
            TESS-specific configuration object.
        global_config : Any
            Global configuration settings.
        logger : logging.Logger, optional
            Logger instance.
        """
        super().__init__(config, global_config, logger)
        self.source_name = "tess"
        
        # Set up output directories
        raw_subdir = self.config.output.get('raw', 'tess')
        processed_subdir = self.config.output.get('processed', 'tess')
        self.raw_output_dir = self.raw_dir / raw_subdir
        self.processed_output_dir = self.processed_dir / processed_subdir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized TESSLoader with config: {config}")
    
    def download(self, tic_id: Optional[int] = None, **kwargs) -> DownloadMetadata:
        """
        Download TESS light curve data for a specific target.
        
        Parameters
        ----------
        tic_id : int, optional
            TESS Input Catalog ID. If None, uses targets from config.
        **kwargs
            Additional arguments:
            - cadence: str, cadence type ('short', 'fast', 'auto')
            - data_product: str, data product type ('SAP', 'PDCSAP')
            - sectors: List[int], specific sectors to download
        
        Returns
        -------
        DownloadMetadata
            Metadata about downloaded data.
        """
        metadata = DownloadMetadata(source=self.source_name)
        start_time = time.time()
        
        # Determine target(s) to download
        if tic_id is None:
            if self.config.targets.tic_ids:
                tic_ids = self.config.targets.tic_ids
            elif self.config.targets.sky_region:
                # Query by sky region
                tic_ids = self._query_by_sky_region(**self.config.targets.sky_region)
            else:
                raise ValueError("No targets specified in config and no tic_id provided")
        else:
            tic_ids = [tic_id]
        
        cadence = kwargs.get('cadence', self.config.cadence)
        data_products = kwargs.get('data_products', self.config.data_products)
        sectors = kwargs.get('sectors', self.config.sectors)
        
        self.logger.info(f"Downloading TESS data for {len(tic_ids)} target(s)")
        
        for tic_id in tic_ids:
            try:
                self.logger.info(f"Downloading TIC {tic_id}")
                
                # Download for each data product
                for data_product in data_products:
                    try:
                        file_paths = self._download_single_target(
                            tic_id, cadence, data_product, sectors=sectors, **kwargs
                        )
                        metadata.file_paths.extend(file_paths)
                        metadata.num_files += len(file_paths)
                    except Exception as e:
                        error_msg = f"Failed to download {data_product} for TIC {tic_id}: {e}"
                        self.logger.error(error_msg)
                        metadata.errors.append(error_msg)
                
                metadata.target_id = str(tic_id)
                
            except Exception as e:
                error_msg = f"Failed to download TIC {tic_id}: {e}"
                self.logger.error(error_msg)
                metadata.errors.append(error_msg)
        
        metadata.download_time = time.time() - start_time
        
        # Calculate total size
        for file_path in metadata.file_paths:
            if Path(file_path).exists():
                metadata.data_size_mb += Path(file_path).stat().st_size / (1024 * 1024)
        
        self.logger.info(
            f"Download complete: {metadata.num_files} files, "
            f"{metadata.data_size_mb:.2f} MB, "
            f"{len(metadata.errors)} errors"
        )
        
        return metadata
    
    def _download_single_target(
        self, 
        tic_id: int, 
        cadence: str, 
        data_product: str,
        sectors: Optional[List[int]] = None,
        **kwargs
    ) -> List[str]:
        """
        Download light curve for a single target.
        
        Parameters
        ----------
        tic_id : int
            TESS Input Catalog ID.
        cadence : str
            Cadence type ('short', 'fast', 'auto').
        data_product : str
            Data product type ('SAP', 'PDCSAP').
        sectors : List[int], optional
            Specific sectors to download. If None, downloads all available.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[str]
            List of paths to downloaded files.
        """
        file_paths = []
        
        def _download_func():
            # Search for light curves
            search_result = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission='TESS',
                cadence=cadence if cadence != 'auto' else 'short'
            )
            
            if len(search_result) == 0:
                raise ValueError(f"No light curves found for TIC {tic_id}")
            
            # Filter by sectors if specified
            if sectors:
                filtered_result = search_result[np.isin(
                    [getattr(lc, 'sector', None) for lc in search_result],
                    sectors
                )]
                if len(filtered_result) == 0:
                    raise ValueError(f"No light curves found for TIC {tic_id} in sectors {sectors}")
                search_result = filtered_result
            
            # Download all available sectors
            lc_collection = search_result.download_all()
            
            # Save raw FITS files
            for i, lc in enumerate(lc_collection):
                # Select data product
                if data_product == 'PDCSAP' and hasattr(lc, 'pdcsap_flux'):
                    flux_col = 'pdcsap_flux'
                elif data_product == 'SAP' and hasattr(lc, 'sap_flux'):
                    flux_col = 'sap_flux'
                else:
                    # Use default flux column
                    flux_col = lc.flux.column_name if hasattr(lc.flux, 'column_name') else 'flux'
                
                # Create filename
                sector = getattr(lc, 'sector', i)
                camera = getattr(lc, 'camera', 1)
                ccd = getattr(lc, 'ccd', 1)
                filename = f"tic_{tic_id:016d}_{data_product}_s{sector:03d}_c{camera}_ccd{ccd}.fits"
                output_path = self.raw_output_dir / filename
                
                # Save raw FITS
                if not self.is_cached(output_path):
                    lc.to_fits(output_path)
                    self.logger.debug(f"Saved raw FITS: {output_path}")
                
                file_paths.append(str(output_path))
            
            return file_paths
        
        # Execute with retry logic
        file_paths = self._retry_with_backoff(_download_func)
        
        return file_paths
    
    def _query_by_sky_region(self, ra: float, dec: float, radius_deg: float) -> List[int]:
        """
        Query TIC IDs by sky region.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius_deg : float
            Search radius in degrees.
        
        Returns
        -------
        List[int]
            List of TIC IDs in the region.
        """
        # This would require querying the TIC catalog
        # For now, raise NotImplementedError
        # In practice, this would use astroquery to query MAST or TIC catalog
        raise NotImplementedError(
            "Sky region queries require TIC catalog access. "
            "Please specify TIC IDs directly in config."
        )
    
    def parse(self, raw_data_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Parse TESS FITS file into structured format.
        
        Parameters
        ----------
        raw_data_path : Path
            Path to raw FITS file.
        **kwargs
            Additional arguments:
            - data_product: str, which data product to extract
        
        Returns
        -------
        Dict[str, Any]
            Parsed data dictionary with keys:
            - 'time': observation times (BTJD)
            - 'flux': flux values
            - 'flux_err': flux uncertainties
            - 'quality': quality flags
            - 'metadata': dictionary of metadata
        """
        data_product = kwargs.get('data_product', 'PDCSAP')
        
        def _parse_func():
            # Load light curve
            lc = lk.read(raw_data_path)
            
            # Extract data product
            if data_product == 'PDCSAP' and hasattr(lc, 'pdcsap_flux'):
                flux = lc.pdcsap_flux.value
                flux_err = lc.pdcsap_flux_err.value if hasattr(lc, 'pdcsap_flux_err') else None
            elif data_product == 'SAP' and hasattr(lc, 'sap_flux'):
                flux = lc.sap_flux.value
                flux_err = lc.sap_flux_err.value if hasattr(lc, 'sap_flux_err') else None
            else:
                flux = lc.flux.value
                flux_err = lc.flux_err.value if hasattr(lc, 'flux_err') else None
            
            # Extract time (BTJD - Barycentric TESS Julian Date)
            time = lc.time.value
            
            # Extract quality flags
            quality = lc.quality.value if hasattr(lc, 'quality') else np.zeros(len(time), dtype=int)
            
            # Apply quality filtering if enabled
            if self.config.quality_filtering:
                # TESS quality flags: 0 = good data
                good_quality = (quality == 0)
                
                time = time[good_quality]
                flux = flux[good_quality]
                if flux_err is not None:
                    flux_err = flux_err[good_quality]
                quality = quality[good_quality]
            
            # Extract metadata
            metadata = {
                'tic_id': getattr(lc, 'targetid', None),
                'mission': 'TESS',
                'sector': getattr(lc, 'sector', None),
                'camera': getattr(lc, 'camera', None),
                'ccd': getattr(lc, 'ccd', None),
                'cadence': getattr(lc, 'cadence', None),
                'data_product': data_product,
                'exptime': getattr(lc, 'exptime', None),
                'ra': getattr(lc, 'ra', None),
                'dec': getattr(lc, 'dec', None),
                'n_points': len(time),
                'time_range': [float(time.min()), float(time.max())] if len(time) > 0 else None,
            }
            
            parsed_data = {
                'time': time,
                'flux': flux,
                'flux_err': flux_err,
                'quality': quality,
                'metadata': metadata
            }
            
            return parsed_data
        
        return self._retry_with_backoff(_parse_func)
    
    def save_raw(self, data: Any, output_path: Path, **kwargs) -> Path:
        """
        Save raw data to disk.
        
        For TESS, raw data is already in FITS format from download.
        This method is mainly for compatibility with the base interface.
        
        Parameters
        ----------
        data : Any
            Raw data (LightCurve object or file-like).
        output_path : Path
            Output file path.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Path
            Path to saved file.
        """
        if hasattr(data, 'to_fits'):
            # LightCurve object
            data.to_fits(output_path)
        else:
            # Assume it's already a file path or file-like object
            with open(output_path, 'wb') as f:
                if hasattr(data, 'read'):
                    f.write(data.read())
                else:
                    f.write(data)
        
        return output_path
    
    def save_processed(self, parsed_data: Dict[str, Any], output_path: Path, **kwargs) -> Path:
        """
        Save processed data to disk as NumPy archive.
        
        Parameters
        ----------
        parsed_data : Dict[str, Any]
            Parsed data dictionary.
        output_path : Path
            Output file path.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Path
            Path to saved file.
        """
        # Ensure .npz extension
        if not output_path.suffix == '.npz':
            output_path = output_path.with_suffix('.npz')
        
        # Save as NumPy archive
        np.savez_compressed(
            output_path,
            time=parsed_data['time'],
            flux=parsed_data['flux'],
            flux_err=parsed_data.get('flux_err'),
            quality=parsed_data.get('quality'),
        )
        
        # Save metadata separately as JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(parsed_data['metadata'], f, indent=2, default=str)
        
        self.logger.debug(f"Saved processed data: {output_path}")
        
        return output_path
    
    def fetch_metadata(self, tic_id: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Fetch metadata about available TESS data.
        
        Parameters
        ----------
        tic_id : int, optional
            TIC ID to query. If None, uses config targets.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with available sectors, cadences, etc.
        """
        if tic_id is None:
            if self.config.targets.tic_ids:
                tic_id = self.config.targets.tic_ids[0]
            else:
                raise ValueError("No tic_id provided and none in config")
        
        def _fetch_func():
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission='TESS')
            
            metadata = {
                'tic_id': tic_id,
                'num_sectors': len(search_result),
                'sectors': [],
                'cadences': [],
                'data_products': [],
                'available': len(search_result) > 0
            }
            
            if len(search_result) > 0:
                # Get sample light curve to check properties
                sample_lc = search_result[0].download()
                metadata['cadences'] = list(set([str(lc.cadence) for lc in search_result]))
                metadata['sectors'] = [
                    getattr(lc, 'sector', i) 
                    for i, lc in enumerate(search_result)
                    if hasattr(lc, 'sector')
                ]
                
                # Check available data products
                if hasattr(sample_lc, 'pdcsap_flux'):
                    metadata['data_products'].append('PDCSAP')
                if hasattr(sample_lc, 'sap_flux'):
                    metadata['data_products'].append('SAP')
            
            return metadata
        
        return self._retry_with_backoff(_fetch_func)
