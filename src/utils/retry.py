from functools import wraps
import time
import logging
from typing import Callable, Type, Tuple, Optional

def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        log.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    wait_time = (backoff_factor ** attempt)
                    log.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
            # This part should theoretically not be reached due to raise above
            if last_exception:
                raise last_exception
        return wrapper
    return decorator
