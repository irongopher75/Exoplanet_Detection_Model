"""
Kepler Space Telescope data downloader.

This module provides functionality to download and process light curve data
from the Kepler Space Telescope mission via the MAST archive.

Data Source:
    - Mission: Kepler Space Telescope (NASA)
    - Archive: MAST (Mikulski Archive for Space Telescopes)
    - Access: Via lightkurve and astroquery packages

Astrophysical Context:
    Kepler observed ~150,000 stars continuously for 4 years, detecting
    thousands of exoplanet transits. The mission provided two main data products:
    - SAP (Simple Aperture Photometry): Raw flux measurements
    - PDCSAP (Pre-search Data Conditioning SAP): Systematics-corrected flux

Assumptions:
    - Targets are identified by KIC (Kepler Input Catalog) IDs
    - Data is available in FITS format from MAST
    - Quality flags follow Kepler standard definitions
    - Missing cadences are handled gracefully

Limitations:
    - Large datasets may require significant download time
    - Some targets may have incomplete coverage
    - Quality flags must be interpreted correctly for science use
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
        "lightkurve is required for Kepler data download. "
        "Install with: pip install lightkurve"
    )

from .base_loader import BaseLoader, DownloadMetadata


class KeplerLoader(BaseLoader):
    """
    Downloader for Kepler Space Telescope light curve data.
    
    This class handles downloading, parsing, and storing Kepler light curves
    from the MAST archive using the lightkurve package.
    """
    
    def __init__(self, config: Any, global_config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize Kepler loader.
        
        Parameters
        ----------
        config : Any
            Kepler-specific configuration object.
        global_config : Any
            Global configuration settings.
        logger : logging.Logger, optional
            Logger instance.
        """
        super().__init__(config, global_config, logger)
        self.source_name = "kepler"
        
        # Set up output directories
        raw_subdir = self.config.output.get('raw', 'kepler')
        processed_subdir = self.config.output.get('processed', 'kepler')
        self.raw_output_dir = self.raw_dir / raw_subdir
        self.processed_output_dir = self.processed_dir / processed_subdir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized KeplerLoader with config: {config}")
    
    def download(self, kic_id: Optional[int] = None, **kwargs) -> DownloadMetadata:
        """
        Download Kepler light curve data for a specific target.
        
        Parameters
        ----------
        kic_id : int, optional
            Kepler Input Catalog ID. If None, uses targets from config.
        **kwargs
            Additional arguments:
            - cadence: str, cadence type ('short', 'long', 'auto')
            - data_product: str, data product type ('SAP', 'PDCSAP')
            - quarter: int, specific quarter to download
        
        Returns
        -------
        DownloadMetadata
            Metadata about downloaded data.
        """
        metadata = DownloadMetadata(source=self.source_name)
        start_time = time.time()
        
        # Determine target(s) to download
        if kic_id is None:
            if self.config.targets.kic_ids:
                kic_ids = self.config.targets.kic_ids
            elif self.config.targets.sky_region:
                # Query by sky region
                kic_ids = self._query_by_sky_region(**self.config.targets.sky_region)
            else:
                raise ValueError("No targets specified in config and no kic_id provided")
        else:
            kic_ids = [kic_id]
        
        cadence = kwargs.get('cadence', self.config.cadence)
        data_products = kwargs.get('data_products', self.config.data_products)
        
        self.logger.info(f"Downloading Kepler data for {len(kic_ids)} target(s)")
        
        for kic_id in kic_ids:
            try:
                self.logger.info(f"Downloading KIC {kic_id}")
                
                # Download for each data product
                for data_product in data_products:
                    try:
                        file_paths = self._download_single_target(
                            kic_id, cadence, data_product, **kwargs
                        )
                        metadata.file_paths.extend(file_paths)
                        metadata.num_files += len(file_paths)
                    except Exception as e:
                        error_msg = f"Failed to download {data_product} for KIC {kic_id}: {e}"
                        self.logger.error(error_msg)
                        metadata.errors.append(error_msg)
                
                metadata.target_id = str(kic_id)
                
            except Exception as e:
                error_msg = f"Failed to download KIC {kic_id}: {e}"
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
        kic_id: int, 
        cadence: str, 
        data_product: str,
        **kwargs
    ) -> List[str]:
        """
        Download light curve for a single target.
        
        Parameters
        ----------
        kic_id : int
            Kepler Input Catalog ID.
        cadence : str
            Cadence type ('short', 'long', 'auto').
        data_product : str
            Data product type ('SAP', 'PDCSAP').
        **kwargs
            Additional arguments (e.g., quarter).
        
        Returns
        -------
        List[str]
            List of paths to downloaded files.
        """
        file_paths = []
        
        def _download_func():
            # Search for light curves
            search_result = lk.search_lightcurve(
                f"KIC {kic_id}",
                mission='Kepler',
                cadence=cadence if cadence != 'auto' else 'short'
            )
            
            if len(search_result) == 0:
                raise ValueError(f"No light curves found for KIC {kic_id}")
            
            # Download all available quarters
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
                quarter = getattr(lc, 'quarter', i)
                filename = f"kic_{kic_id:09d}_{data_product}_q{quarter:02d}.fits"
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
        Query KIC IDs by sky region.
        
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
            List of KIC IDs in the region.
        """
        # This would require querying the KIC catalog
        # For now, raise NotImplementedError
        # In practice, this would use astroquery to query MAST or KIC catalog
        raise NotImplementedError(
            "Sky region queries require KIC catalog access. "
            "Please specify KIC IDs directly in config."
        )
    
    def parse(self, raw_data_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Parse Kepler FITS file into structured format.
        
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
            - 'time': observation times (BJD - 2454833)
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
            
            # Extract time (convert to BJD if needed)
            time = lc.time.value
            
            # Extract quality flags
            quality = lc.quality.value if hasattr(lc, 'quality') else np.zeros(len(time), dtype=int)
            
            # Apply quality filtering if enabled
            if self.config.quality_filtering:
                if self.config.quality_flags:
                    # Use custom quality flags
                    good_quality = np.isin(quality, self.config.quality_flags, invert=True)
                else:
                    # Use default: exclude bad data (quality != 0)
                    good_quality = (quality == 0)
                
                time = time[good_quality]
                flux = flux[good_quality]
                if flux_err is not None:
                    flux_err = flux_err[good_quality]
                quality = quality[good_quality]
            
            # Extract metadata
            metadata = {
                'kic_id': getattr(lc, 'targetid', None),
                'mission': 'Kepler',
                'quarter': getattr(lc, 'quarter', None),
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
        
        For Kepler, raw data is already in FITS format from download.
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
    
    def fetch_metadata(self, kic_id: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Fetch metadata about available Kepler data.
        
        Parameters
        ----------
        kic_id : int, optional
            KIC ID to query. If None, uses config targets.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with available quarters, cadences, etc.
        """
        if kic_id is None:
            if self.config.targets.kic_ids:
                kic_id = self.config.targets.kic_ids[0]
            else:
                raise ValueError("No kic_id provided and none in config")
        
        def _fetch_func():
            search_result = lk.search_lightcurve(f"KIC {kic_id}", mission='Kepler')
            
            metadata = {
                'kic_id': kic_id,
                'num_quarters': len(search_result),
                'quarters': [],
                'cadences': [],
                'data_products': [],
                'available': len(search_result) > 0
            }
            
            if len(search_result) > 0:
                # Get sample light curve to check properties
                sample_lc = search_result[0].download()
                metadata['cadences'] = list(set([str(lc.cadence) for lc in search_result]))
                metadata['quarters'] = [getattr(lc, 'quarter', i) for i, lc in enumerate(search_result)]
                
                # Check available data products
                if hasattr(sample_lc, 'pdcsap_flux'):
                    metadata['data_products'].append('PDCSAP')
                if hasattr(sample_lc, 'sap_flux'):
                    metadata['data_products'].append('SAP')
            
            return metadata
        
        return self._retry_with_backoff(_fetch_func)
