"""
Gaia stellar data loader.

This module provides functionality to download stellar parameters from
the Gaia archive via astroquery.

Data Source:
    - Mission: Gaia (European Space Agency)
    - Archive: Gaia Archive
    - Access: Via astroquery.gaia package

Astrophysical Context:
    Gaia provides precise astrometry, photometry, and spectroscopy for
    over 1 billion stars. Stellar parameters are essential for:
    - Calculating planet radii from transit depths
    - Determining stellar masses for RV analysis
    - Understanding stellar evolution context
    - Cross-matching between different surveys

Assumptions:
    - Targets are matched by sky coordinates (RA, Dec)
    - Cross-matching radius is configurable
    - Stellar parameters are available in Gaia DR3 or later
    - Coordinates are in ICRS frame

Limitations:
    - Large queries may be slow
    - Some parameters may be missing for faint stars
    - Cross-matching requires accurate coordinates
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

try:
    from astroquery.gaia import Gaia
except ImportError:
    raise ImportError(
        "astroquery is required for Gaia data download. "
        "Install with: pip install astroquery"
    )

from .base_loader import BaseLoader, DownloadMetadata


class GaiaLoader(BaseLoader):
    """
    Downloader for Gaia stellar data.
    
    This class handles querying the Gaia archive to fetch stellar parameters
    for targets identified by coordinates.
    """
    
    def __init__(self, config: Any, global_config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize Gaia loader.
        
        Parameters
        ----------
        config : Any
            Gaia-specific configuration object.
        global_config : Any
            Global configuration settings.
        logger : logging.Logger, optional
            Logger instance.
        """
        super().__init__(config, global_config, logger)
        self.source_name = "gaia"
        
        # Initialize Gaia query interface
        self.gaia = Gaia()
        
        # Set up output directories
        processed_subdir = self.config.output.get('processed', 'stellar_context')
        self.processed_output_dir = self.processed_dir / processed_subdir
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.crossmatch_radius = self.config.crossmatch_radius_arcsec / 3600.0  # Convert to degrees
        self.parameters = self.config.parameters
        
        self.logger.info(f"Initialized GaiaLoader with crossmatch radius: {self.crossmatch_radius*3600:.2f} arcsec")
    
    def download(self, coordinates: Optional[List[Tuple[float, float]]] = None, **kwargs) -> DownloadMetadata:
        """
        Download Gaia stellar data for target(s).
        
        Parameters
        ----------
        coordinates : List[Tuple[float, float]], optional
            List of (RA, Dec) tuples in degrees. If None, must be provided
            via other means (e.g., from photometry data).
        **kwargs
            Additional arguments:
            - target_ids: List[str], specific Gaia source IDs
        
        Returns
        -------
        DownloadMetadata
            Metadata about downloaded data.
        """
        metadata = DownloadMetadata(source=self.source_name)
        start_time = time.time()
        
        # Determine targets
        if coordinates is None:
            # Try to get coordinates from other sources
            # This could be extended to read from photometry metadata
            raise ValueError("Coordinates must be provided for Gaia queries")
        
        self.logger.info(f"Downloading Gaia data for {len(coordinates)} target(s)")
        
        try:
            # Query Gaia for all targets
            file_paths = self._query_gaia(coordinates, **kwargs)
            metadata.file_paths.extend(file_paths)
            metadata.num_files += len(file_paths)
            
        except Exception as e:
            error_msg = f"Failed to download Gaia data: {e}"
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
    
    def _query_gaia(
        self, 
        coordinates: List[Tuple[float, float]],
        **kwargs
    ) -> List[str]:
        """
        Query Gaia archive for stellar parameters.
        
        Parameters
        ----------
        coordinates : List[Tuple[float, float]]
            List of (RA, Dec) tuples in degrees.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[str]
            List of paths to saved files.
        """
        file_paths = []
        
        def _query_func():
            # Build ADQL query
            # For multiple targets, we use a cone search for each
            # In practice, this could be optimized with a single query
            
            all_results = []
            
            for ra, dec in coordinates:
                # Build parameter list
                if self.parameters:
                    param_list = ', '.join(self.parameters)
                else:
                    # Default parameters
                    param_list = """
                        source_id, ra, dec, parallax, parallax_error,
                        pmra, pmdec, teff_val, teff_percentile_lower, teff_percentile_upper,
                        radius_val, radius_percentile_lower, radius_percentile_upper,
                        lum_val, lum_percentile_lower, lum_percentile_upper
                    """
                
                # ADQL query for cone search
                query = f"""
                    SELECT {param_list}
                    FROM gaiadr3.gaia_source
                    WHERE 1=CONTAINS(
                        POINT('ICRS', ra, dec),
                        CIRCLE('ICRS', {ra}, {dec}, {self.crossmatch_radius})
                    )
                    ORDER BY DISTANCE(
                        POINT('ICRS', ra, dec),
                        POINT('ICRS', {ra}, {dec})
                    ) ASC
                """
                
                self.logger.debug(f"Querying Gaia for ({ra}, {dec})")
                
                # Execute query
                job = self.gaia.launch_job(query)
                result = job.get_results()
                
                if len(result) > 0:
                    # Take closest match
                    all_results.append(result[0])
                else:
                    self.logger.warning(f"No Gaia match found for ({ra}, {dec})")
            
            if len(all_results) == 0:
                raise ValueError("No Gaia matches found for any target")
            
            # Combine results into a table
            if len(all_results) == 1:
                df = all_results[0].to_pandas()
            else:
                # Combine multiple results
                dfs = [r.to_pandas() for r in all_results]
                df = pd.concat(dfs, ignore_index=True)
            
            # Save to file
            filename = "gaia_stellar_parameters.csv"
            output_path = self.processed_output_dir / filename
            df.to_csv(output_path, index=False)
            
            self.logger.debug(f"Saved Gaia data: {output_path}")
            file_paths.append(str(output_path))
            
            return file_paths
        
        # Execute with retry logic
        file_paths = self._retry_with_backoff(_query_func)
        
        return file_paths
    
    def parse(self, raw_data_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Parse Gaia CSV into structured format.
        
        Parameters
        ----------
        raw_data_path : Path
            Path to CSV file.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Dict[str, Any]
            Parsed data dictionary with DataFrame and metadata.
        """
        def _parse_func():
            # Read CSV
            df = pd.read_csv(raw_data_path)
            
            # Extract metadata
            metadata = {
                'source': 'Gaia',
                'n_stars': len(df),
                'columns': list(df.columns),
                'download_time': time.time(),
            }
            
            # Check for key parameters
            if 'source_id' in df.columns:
                metadata['has_source_ids'] = True
            if 'teff_val' in df.columns:
                metadata['has_temperature'] = True
            if 'radius_val' in df.columns:
                metadata['has_radius'] = True
            if 'parallax' in df.columns:
                metadata['has_parallax'] = True
            
            parsed_data = {
                'dataframe': df,
                'metadata': metadata
            }
            
            return parsed_data
        
        return self._retry_with_backoff(_parse_func)
    
    def save_raw(self, data: Any, output_path: Path, **kwargs) -> Path:
        """
        Save raw data to disk.
        
        For Gaia, raw data is already in CSV format from query.
        
        Parameters
        ----------
        data : Any
            Raw data (DataFrame, CSV string, or file-like).
        output_path : Path
            Output file path.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Path
            Path to saved file.
        """
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        elif isinstance(data, str):
            with open(output_path, 'w') as f:
                f.write(data)
        elif hasattr(data, 'read'):
            with open(output_path, 'wb') as f:
                f.write(data.read())
        else:
            raise ValueError(f"Cannot save data of type {type(data)}")
        
        return output_path
    
    def save_processed(self, parsed_data: Dict[str, Any], output_path: Path, **kwargs) -> Path:
        """
        Save processed data to disk.
        
        Parameters
        ----------
        parsed_data : Dict[str, Any]
            Parsed data dictionary with 'dataframe' and 'metadata' keys.
        output_path : Path
            Output file path.
        **kwargs
            Additional arguments:
            - format: str, output format ('csv', 'parquet', 'hdf5')
        
        Returns
        -------
        Path
            Path to saved file.
        """
        format_type = kwargs.get('format', 'csv')
        df = parsed_data['dataframe']
        
        if format_type == 'csv':
            if not output_path.suffix == '.csv':
                output_path = output_path.with_suffix('.csv')
            df.to_csv(output_path, index=False)
        elif format_type == 'parquet':
            if not output_path.suffix == '.parquet':
                output_path = output_path.with_suffix('.parquet')
            df.to_parquet(output_path, index=False)
        elif format_type == 'hdf5':
            if not output_path.suffix == '.h5':
                output_path = output_path.with_suffix('.h5')
            df.to_hdf(output_path, key='gaia', mode='w')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Save metadata separately
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(parsed_data['metadata'], f, indent=2, default=str)
        
        self.logger.debug(f"Saved processed data: {output_path}")
        
        return output_path
    
    def fetch_metadata(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch metadata about Gaia archive.
        
        Parameters
        ----------
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with available tables, columns, etc.
        """
        def _fetch_func():
            # Query for table information
            # This is a simple test query to verify access
            test_query = """
                SELECT TOP 1 source_id, ra, dec
                FROM gaiadr3.gaia_source
            """
            
            try:
                job = self.gaia.launch_job(test_query)
                result = job.get_results()
                
                metadata = {
                    'archive': 'Gaia',
                    'accessible': True,
                    'data_release': 'DR3',
                    'sample_columns': list(result.columns) if len(result) > 0 else [],
                }
            except Exception as e:
                metadata = {
                    'archive': 'Gaia',
                    'accessible': False,
                    'error': str(e),
                }
            
            return metadata
        
        return self._retry_with_backoff(_fetch_func)

