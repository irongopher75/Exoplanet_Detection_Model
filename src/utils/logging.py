import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_file: str = "training.log"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_dir : Path, optional
        Directory to save log files.
    log_level : str
        Logging level (INFO, DEBUG, etc.).
    log_file : str
        Name of the log file.
    
    Returns
    -------
    logging.Logger
        Root logger instance.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
