"""
Base loader interface for all data downloaders.

This module defines the abstract base class that all dataset-specific downloaders
must inherit from. It enforces a consistent interface and provides common
functionality for data fetching, parsing, and storage.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import time
from dataclasses import dataclass


@dataclass
class DownloadMetadata:
    """Metadata about a downloaded dataset."""
    source: str
    target_id: Optional[str] = None
    download_time: Optional[float] = None
    file_paths: List[str] = None
    num_files: int = 0
    data_size_mb: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.file_paths is None:
            self.file_paths = []
        if self.errors is None:
            self.errors = []
        if self.download_time is None:
            self.download_time = time.time()


class BaseLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    All dataset-specific downloaders must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across all data sources.
    
    Attributes
    ----------
    config : Any
        Configuration object for this loader.
    global_config : Any
        Global configuration settings.
    logger : logging.Logger
        Logger instance for this loader.
    data_root : Path
        Root directory for data storage.
    raw_dir : Path
        Directory for raw data files.
    processed_dir : Path
        Directory for processed data files.
    metadata_dir : Path
        Directory for metadata files.
    """
    
    def __init__(self, config: Any, global_config: Any, logger: Optional[logging.Logger] = None):
        """
        Initialize the base loader.
        
        Parameters
        ----------
        config : Any
            Configuration object specific to this loader.
        global_config : Any
            Global configuration settings.
        logger : logging.Logger, optional
            Logger instance. If None, a new logger will be created.
        """
        self.config = config
        self.global_config = global_config
        self.data_root = Path(global_config.data_root)
        self.raw_dir = Path(global_config.raw_dir)
        self.processed_dir = Path(global_config.processed_dir)
        self.metadata_dir = Path(global_config.metadata_dir)
        
        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def download(self, **kwargs) -> DownloadMetadata:
        """
        Download raw data from the source.
        
        This method should handle:
        - Network requests
        - Retry logic
        - Progress reporting
        - Error handling
        
        Parameters
        ----------
        **kwargs
            Additional arguments specific to the downloader.
        
        Returns
        -------
        DownloadMetadata
            Metadata about the downloaded data.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement download()")
    
    @abstractmethod
    def parse(self, raw_data_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Parse raw data files into structured format.
        
        This method should:
        - Read raw data files (FITS, CSV, etc.)
        - Extract time series, metadata, etc.
        - Handle missing or corrupted data
        - Normalize formats
        
        Parameters
        ----------
        raw_data_path : Path
            Path to raw data file(s).
        **kwargs
            Additional arguments for parsing.
        
        Returns
        -------
        Dict[str, Any]
            Parsed data dictionary with keys like:
            - 'time': array of observation times
            - 'flux': array of flux values
            - 'flux_err': array of flux uncertainties
            - 'quality': array of quality flags
            - 'metadata': dictionary of metadata
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement parse()")
    
    @abstractmethod
    def save_raw(self, data: Any, output_path: Path, **kwargs) -> Path:
        """
        Save raw data to disk.
        
        Parameters
        ----------
        data : Any
            Raw data to save (could be file-like object, bytes, etc.).
        output_path : Path
            Path where raw data should be saved.
        **kwargs
            Additional arguments for saving.
        
        Returns
        -------
        Path
            Path to saved file.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement save_raw()")
    
    @abstractmethod
    def save_processed(self, parsed_data: Dict[str, Any], output_path: Path, **kwargs) -> Path:
        """
        Save processed/parsed data to disk.
        
        Parameters
        ----------
        parsed_data : Dict[str, Any]
            Parsed data dictionary from parse().
        output_path : Path
            Path where processed data should be saved.
        **kwargs
            Additional arguments for saving.
        
        Returns
        -------
        Path
            Path to saved file.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement save_processed()")
    
    @abstractmethod
    def fetch_metadata(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch metadata about available data.
        
        This method should query the data source to determine:
        - What targets are available
        - What time ranges are covered
        - What data products exist
        - File sizes, etc.
        
        Parameters
        ----------
        **kwargs
            Additional arguments for metadata fetching.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fetch_metadata()")
    
    def _retry_with_backoff(self, func, *args, max_attempts: Optional[int] = None, **kwargs):
        """
        Execute a function with exponential backoff retry logic.
        
        Parameters
        ----------
        func : callable
            Function to execute.
        *args
            Positional arguments for func.
        max_attempts : int, optional
            Maximum number of retry attempts. If None, uses config default.
        **kwargs
            Keyword arguments for func.
        
        Returns
        -------
        Any
            Return value of func.
        
        Raises
        ------
        Exception
            If all retry attempts fail, raises the last exception.
        """
        if max_attempts is None:
            max_attempts = self.global_config.retry.max_attempts
        
        backoff_factor = self.global_config.retry.backoff_factor
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    wait_time = backoff_factor ** attempt
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_attempts} attempts failed")
        
        raise last_exception
    
    def _get_output_path(self, subdir: str, filename: str, create: bool = True) -> Path:
        """
        Get output path for a file in a subdirectory.
        
        Parameters
        ----------
        subdir : str
            Subdirectory name (relative to appropriate root).
        filename : str
            Filename.
        create : bool
            Whether to create the directory if it doesn't exist.
        
        Returns
        -------
        Path
            Full path to output file.
        """
        if subdir:
            output_dir = self.data_root / subdir
        else:
            output_dir = self.data_root
        
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir / filename
    
    def is_cached(self, file_path: Path) -> bool:
        """
        Check if a file already exists (cached).
        
        Parameters
        ----------
        file_path : Path
            Path to check.
        
        Returns
        -------
        bool
            True if file exists, False otherwise.
        """
        return file_path.exists() and file_path.stat().st_size > 0
