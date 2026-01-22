"""
NASA Exoplanet Archive data fetcher.

This module provides functionality to download exoplanet catalog data from
the NASA Exoplanet Archive via their REST API.

Data Source:
    - Archive: NASA Exoplanet Archive (IPAC/Caltech)
    - Access: Public REST API
    - URL: https://exoplanetarchive.ipac.caltech.edu/

Astrophysical Context:
    The NASA Exoplanet Archive is the official repository for confirmed
    exoplanets and their properties. It includes:
    - Confirmed planets with orbital parameters
    - False positives (e.g., eclipsing binaries)
    - Stellar parameters
    - Discovery methods and references

Assumptions:
    - API is publicly accessible
    - Data is returned in CSV format
    - Column names are standardized
    - Updates are periodic but not real-time

Limitations:
    - Large tables may require pagination
    - Some columns may be missing for certain planets
    - API rate limits may apply
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time
import requests
from urllib.parse import urlencode

from .base_loader import BaseLoader, DownloadMetadata


class ExoplanetArchiveLoader(BaseLoader):
    """
    Downloader for NASA Exoplanet Archive data.
    
    This class handles fetching exoplanet catalog data from the NASA
    Exoplanet Archive REST API.
    """
    
    def __init__(self, config: Any, global_config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize Exoplanet Archive loader.
        
        Parameters
        ----------
        config : Any
            Exoplanet Archive-specific configuration object.
        global_config : Any
            Global configuration settings.
        logger : logging.Logger, optional
            Logger instance.
        """
        super().__init__(config, global_config, logger)
        self.source_name = "exoplanet_archive"
        self.api_base_url = config.api_base_url
        
        # Set up output directories
        processed_subdir = self.config.output.get('processed', 'exoplanet_labels')
        self.processed_output_dir = self.processed_dir / processed_subdir
        self.processed_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized ExoplanetArchiveLoader with API: {self.api_base_url}")
    
    def download(self, table_name: Optional[str] = None, **kwargs) -> DownloadMetadata:
        """
        Download exoplanet catalog data from NASA Exoplanet Archive.
        
        Parameters
        ----------
        table_name : str, optional
            Specific table to download. If None, downloads all tables from config.
        **kwargs
            Additional arguments:
            - columns: List[str], specific columns to fetch
            - filters: Dict, query filters
        
        Returns
        -------
        DownloadMetadata
            Metadata about downloaded data.
        """
        metadata = DownloadMetadata(source=self.source_name)
        start_time = time.time()
        
        # Determine which tables to download
        if table_name:
            tables_to_download = [
                {'name': table_name, **kwargs}
            ]
        else:
            tables_to_download = self.config.tables
        
        if not tables_to_download:
            raise ValueError("No tables specified in config")
        
        self.logger.info(f"Downloading {len(tables_to_download)} table(s) from Exoplanet Archive")
        
        for table_config in tables_to_download:
            try:
                table_name = table_config.get('name', 'exoplanets')
                self.logger.info(f"Downloading table: {table_name}")
                
                file_paths = self._download_table(table_config)
                metadata.file_paths.extend(file_paths)
                metadata.num_files += len(file_paths)
                
            except Exception as e:
                error_msg = f"Failed to download table {table_name}: {e}"
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
    
    def _download_table(self, table_config: Dict[str, Any]) -> List[str]:
        """
        Download a single table from the Exoplanet Archive.
        
        Parameters
        ----------
        table_config : Dict[str, Any]
            Table configuration with name, endpoint, columns, filters, etc.
        
        Returns
        -------
        List[str]
            List of paths to downloaded files.
        """
        file_paths = []
        
        def _download_func():
            table_name = table_config.get('name', 'exoplanets')
            endpoint = table_config.get('endpoint', 'exoplanets')
            columns = table_config.get('columns', [])
            filters = table_config.get('filters', {})
            format_type = table_config.get('format', 'csv')
            
            # Build API URL
            params = {
                'table': endpoint,
                'format': format_type
            }
            
            # Add column selection if specified
            if columns:
                params['select'] = ','.join(columns)
            
            # Add filters if specified
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        where_clauses.append(f"{key} IN ({','.join(map(str, value))})")
                    else:
                        where_clauses.append(f"{key} = {value}")
                if where_clauses:
                    params['where'] = ' AND '.join(where_clauses)
            
            url = f"{self.api_base_url}?{urlencode(params)}"
            
            self.logger.debug(f"Fetching from URL: {url}")
            
            # Make request with timeout
            response = requests.get(
                url,
                timeout=self.global_config.retry.timeout_seconds
            )
            response.raise_for_status()
            
            # Save raw CSV
            filename = f"{table_name}.csv"
            output_path = self.processed_output_dir / filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.debug(f"Saved table: {output_path}")
            file_paths.append(str(output_path))
            
            return file_paths
        
        # Execute with retry logic
        file_paths = self._retry_with_backoff(_download_func)
        
        return file_paths
    
    def parse(self, raw_data_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Parse Exoplanet Archive CSV into structured format.
        
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
                'source': 'NASA Exoplanet Archive',
                'table_name': raw_data_path.stem,
                'n_planets': len(df),
                'columns': list(df.columns),
                'download_time': time.time(),
            }
            
            # Check for common columns
            if 'pl_name' in df.columns:
                metadata['has_planet_names'] = True
            if 'hostname' in df.columns:
                metadata['has_host_names'] = True
            if 'discoverymethod' in df.columns:
                metadata['discovery_methods'] = df['discoverymethod'].unique().tolist()
            
            parsed_data = {
                'dataframe': df,
                'metadata': metadata
            }
            
            return parsed_data
        
        return self._retry_with_backoff(_parse_func)
    
    def save_raw(self, data: Any, output_path: Path, **kwargs) -> Path:
        """
        Save raw data to disk.
        
        For Exoplanet Archive, raw data is CSV format.
        
        Parameters
        ----------
        data : Any
            Raw data (bytes, string, or file-like).
        output_path : Path
            Output file path.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        Path
            Path to saved file.
        """
        if isinstance(data, (bytes, str)):
            with open(output_path, 'wb' if isinstance(data, bytes) else 'w') as f:
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
            df.to_hdf(output_path, key='exoplanets', mode='w')
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
        Fetch metadata about available Exoplanet Archive tables.
        
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
            # The Exoplanet Archive API doesn't have a direct metadata endpoint,
            # so we fetch a small sample to infer structure
            
            test_url = f"{self.api_base_url}?table=exoplanets&format=csv&select=pl_name&where=pl_name+like+'%'&limit=1"
            
            response = requests.get(
                test_url,
                timeout=self.global_config.retry.timeout_seconds
            )
            response.raise_for_status()
            
            # Parse CSV header
            lines = response.text.strip().split('\n')
            if len(lines) > 0:
                columns = lines[0].split(',')
            else:
                columns = []
            
            metadata = {
                'api_base_url': self.api_base_url,
                'available_tables': ['exoplanets', 'composite'],
                'sample_columns': columns,
                'format': 'csv',
                'accessible': True
            }
            
            return metadata
        
        return self._retry_with_backoff(_fetch_func)

