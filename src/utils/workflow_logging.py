import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_workflow_logging(
    workflow_name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """Unified logging setup for all workflows."""
    logger = logging.getLogger(workflow_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"{workflow_name}_{timestamp}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
