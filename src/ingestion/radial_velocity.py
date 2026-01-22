"""
Radial Velocity (RV) data downloader.

This module provides functionality to download radial velocity data from
ESO (European Southern Observatory) archives, including HARPS and ESPRESSO.

Data Source:
    - Archives: ESO Phase-3 Archive
    - Instruments: HARPS, ESPRESSO, and others
    - Access: Via astroquery.eso package

Astrophysical Context:
    Radial velocity measurements detect exoplanets by measuring the Doppler
    shift of stellar spectral lines. As a planet orbits, it causes the star
    to wobble, creating a periodic RV signal. RV data complements transit
    photometry by providing planet masses and orbital eccentricities.

Assumptions:
    - Targets are identified by star names or coordinates
    - Data is available in FITS format from ESO archives
    - RV measurements include uncertainties
    - Timestamps are normalized to a common reference frame

Limitations:
    - ESO archive access may require registration for some datasets
    - Not all targets have RV data available
    - Data formats may vary between instruments
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

try:
    from astroquery.eso import Eso
except ImportError:
    raise ImportError(
        "astroquery is required for RV data download. "
        "Install with: pip install astroquery"
    )

from .base_loader import BaseLoader, DownloadMetadata


class RadialVelocityLoader(BaseLoader):
    """
    Downloader for radial velocity data from ESO archives.
    
    This class handles downloading, parsing, and storing RV data from
    ESO Phase-3 archives using astroquery.
    """
    
    def __init__(self, config: Any, global_config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize Radial Velocity loader.
        
        Parameters
        ----------
        config : Any
            Radial velocity-specific configuration object.
        global_config : Any
            Global configuration settings.
        logger : logging.Logger, optional
            Logger instance.
        """
        super().__init__(config, global_config, logger)
        self.source_name = "radial_velocity"
        
        # Initialize ESO query interface
        self.eso = Eso()
        # Note: For some datasets, login may be required:
        # self.eso.login('username')
        
        # Set up output directories
        raw_subdir = self.config.output.get('raw', 'radial_velocity')
        processed_subdir = self.config.output.get('processed', 'radial_velocity')
        self.raw_output_dir = self.raw_dir / raw_subdir
        self.processed_output_dir = self.processed_dir / processed_subdir
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized RadialVelocityLoader")
    
    def download(self, target_name: Optional[str] = None, **kwargs) -> DownloadMetadata:
        """
        Download radial velocity data for target(s).
        
        Parameters
        ----------
        target_name : str, optional
            Star name or identifier. If None, uses targets from config.
        **kwargs
            Additional arguments:
            - instrument: str, specific instrument ('HARPS', 'ESPRESSO')
            - coordinates: tuple, (ra, dec) in degrees
        
        Returns
        -------
        DownloadMetadata
            Metadata about downloaded data.
        """
        metadata = DownloadMetadata(source=self.source_name)
        start_time = time.time()
        
        # Determine target(s) to download
        if target_name is None:
            if self.config.targets.star_names:
                targets = self.config.targets.star_names
            elif self.config.targets.coordinates:
                targets = self.config.targets.coordinates
            else:
                raise ValueError("No targets specified in config and no target_name provided")
        else:
            targets = [target_name]
        
        # Get enabled instruments
        enabled_instruments = [
            inst['name'] for inst in self.config.instruments
            if inst.get('enabled', True)
        ]
        
        if not enabled_instruments:
            raise ValueError("No instruments enabled in config")
        
        self.logger.info(f"Downloading RV data for {len(targets)} target(s) using {enabled_instruments}")
        
        for target in targets:
            try:
                if isinstance(target, (list, tuple)):
                    # Coordinates
                    ra, dec = target
                    self.logger.info(f"Downloading RV data for coordinates ({ra}, {dec})")
                else:
                    # Star name
                    self.logger.info(f"Downloading RV data for {target}")
                
                for instrument in enabled_instruments:
                    try:
                        file_paths = self._download_single_target(
                            target, instrument, **kwargs
                        )
                        metadata.file_paths.extend(file_paths)
                        metadata.num_files += len(file_paths)
                    except Exception as e:
                        error_msg = f"Failed to download {instrument} data for {target}: {e}"
                        self.logger.error(error_msg)
                        metadata.errors.append(error_msg)
                
                metadata.target_id = str(target)
                
            except Exception as e:
                error_msg = f"Failed to download RV data for {target}: {e}"
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
        target: Any, 
        instrument: str,
        **kwargs
    ) -> List[str]:
        """
        Download RV data for a single target and instrument.
        
        Parameters
        ----------
        target : Any
            Target identifier (star name or coordinates).
        instrument : str
            Instrument name ('HARPS', 'ESPRESSO', etc.).
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[str]
            List of paths to downloaded files.
        """
        file_paths = []
        
        def _download_func():
            # Query ESO archive
            if isinstance(target, (list, tuple)):
                ra, dec = target
                query_result = self.eso.query_instrument(
                    instrument,
                    target=f"{ra} {dec}",
                    coord_system='equatorial'
                )
            else:
                query_result = self.eso.query_instrument(
                    instrument,
                    target=target
                )
            
            if query_result is None or len(query_result) == 0:
                raise ValueError(f"No RV data found for {target} with {instrument}")
            
            # Download files
            for row in query_result:
                try:
                    # Get file URL
                    file_url = row.get('URL', None)
                    if file_url is None:
                        continue
                    
                    # Create filename
                    obs_id = row.get('DP.ID', f"obs_{len(file_paths)}")
                    filename = f"{instrument.lower()}_{obs_id}.fits"
                    output_path = self.raw_output_dir / filename
                    
                    # Download if not cached
                    if not self.is_cached(output_path):
                        self.eso.retrieve_data([file_url], output_path)
                        self.logger.debug(f"Downloaded: {output_path}")
                    
                    file_paths.append(str(output_path))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to download file for {target}: {e}")
                    continue
            
            return file_paths
        
        # Execute with retry logic
        file_paths = self._retry_with_backoff(_download_func)
        
        return file_paths
    
    def parse(self, raw_data_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Parse RV FITS file into structured format.
        
        Parameters
        ----------
        raw_data_path : Path
            Path to raw FITS file.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Dict[str, Any]
            Parsed data dictionary with keys:
            - 'time': observation times (JD or BJD)
            - 'rv': radial velocity values (m/s)
            - 'rv_err': RV uncertainties (m/s)
            - 'metadata': dictionary of metadata
        """
        from astropy.io import fits
        
        def _parse_func():
            # Read FITS file
            with fits.open(raw_data_path) as hdul:
                # Extract RV data (format varies by instrument)
                # This is a simplified parser - actual format depends on instrument
                
                # Try to find RV data in various extensions
                rv_data = None
                time_data = None
                rv_err_data = None
                
                for hdu in hdul:
                    if hdu.data is not None:
                        # Check for common column names
                        if hasattr(hdu.data, 'dtype') and hdu.data.dtype.names:
                            colnames = hdu.data.dtype.names
                            
                            # Look for RV columns
                            rv_cols = [c for c in colnames if 'rv' in c.lower() or 'vel' in c.lower()]
                            time_cols = [c for c in colnames if 'time' in c.lower() or 'jd' in c.lower() or 'bjd' in c.lower()]
                            err_cols = [c for c in colnames if 'err' in c.lower() or 'e_rv' in c.lower()]
                            
                            if rv_cols:
                                rv_data = hdu.data[rv_cols[0]]
                            if time_cols:
                                time_data = hdu.data[time_cols[0]]
                            if err_cols:
                                rv_err_data = hdu.data[err_cols[0]]
                
                # If no structured data found, try header keywords
                if rv_data is None:
                    header = hdul[0].header
                    # Some RV files store data in header keywords
                    # This is instrument-specific and may need customization
                    self.logger.warning(f"Could not parse RV data from {raw_data_path}")
                    rv_data = np.array([])
                    time_data = np.array([])
                    rv_err_data = np.array([])
                
                # Extract metadata from header
                header = hdul[0].header
                metadata = {
                    'source': 'ESO',
                    'instrument': header.get('INSTRUME', 'unknown'),
                    'target': header.get('OBJECT', 'unknown'),
                    'ra': header.get('RA', None),
                    'dec': header.get('DEC', None),
                    'obs_date': header.get('DATE-OBS', None),
                    'exptime': header.get('EXPTIME', None),
                    'n_points': len(rv_data) if rv_data is not None else 0,
                }
            
            parsed_data = {
                'time': time_data if time_data is not None else np.array([]),
                'rv': rv_data if rv_data is not None else np.array([]),
                'rv_err': rv_err_data if rv_err_data is not None else np.array([]),
                'metadata': metadata
            }
            
            return parsed_data
        
        return self._retry_with_backoff(_parse_func)
    
    def save_raw(self, data: Any, output_path: Path, **kwargs) -> Path:
        """
        Save raw data to disk.
        
        For RV data, raw files are FITS format from ESO.
        
        Parameters
        ----------
        data : Any
            Raw data (file-like object or bytes).
        output_path : Path
            Output file path.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Path
            Path to saved file.
        """
        if hasattr(data, 'read'):
            with open(output_path, 'wb') as f:
                f.write(data.read())
        elif isinstance(data, (bytes, str)):
            with open(output_path, 'wb' if isinstance(data, bytes) else 'w') as f:
                f.write(data)
        else:
            raise ValueError(f"Cannot save data of type {type(data)}")
        
        return output_path
    
    def save_processed(self, parsed_data: Dict[str, Any], output_path: Path, **kwargs) -> Path:
        """
        Save processed data to disk.
        
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
            rv=parsed_data['rv'],
            rv_err=parsed_data.get('rv_err'),
        )
        
        # Save metadata separately
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(parsed_data['metadata'], f, indent=2, default=str)
        
        self.logger.debug(f"Saved processed data: {output_path}")
        
        return output_path
    
    def fetch_metadata(self, target_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Fetch metadata about available RV data.
        
        Parameters
        ----------
        target_name : str, optional
            Target to query. If None, uses config targets.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with available instruments, observations, etc.
        """
        if target_name is None:
            if self.config.targets.star_names:
                target_name = self.config.targets.star_names[0]
            else:
                raise ValueError("No target_name provided and none in config")
        
        def _fetch_func():
            # Query for available data
            query_result = self.eso.query_instrument('HARPS', target=target_name)
            
            metadata = {
                'target': target_name,
                'available_instruments': [],
                'num_observations': 0,
                'available': False
            }
            
            if query_result is not None and len(query_result) > 0:
                metadata['available'] = True
                metadata['num_observations'] = len(query_result)
                metadata['available_instruments'].append('HARPS')
            
            # Check other instruments
            for instrument in ['ESPRESSO']:
                try:
                    result = self.eso.query_instrument(instrument, target=target_name)
                    if result is not None and len(result) > 0:
                        metadata['available_instruments'].append(instrument)
                except:
                    pass
            
            return metadata
        
        return self._retry_with_backoff(_fetch_func)

