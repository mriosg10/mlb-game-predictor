"""
Retry decorator with exponential backoff (NFR-03).
Default: 3 retries at 2s / 4s / 8s.
"""

import logging
import time
from functools import wraps
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_backoff(
    retries: int = 3,
    backoff_base: int = 2,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator that retries a function on the specified exception types.

    delay = backoff_base ** (attempt + 1):
        attempt 0 -> backoff_base^1 = 2s
        attempt 1 -> backoff_base^2 = 4s
        attempt 2 -> backoff_base^3 = 8s
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == retries:
                        logger.error(
                            "%s failed after %d attempt(s): %s",
                            func.__qualname__, retries + 1, exc,
                        )
                        raise
                    delay = backoff_base ** (attempt + 1)
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %ds",
                        func.__qualname__, attempt + 1, retries, exc, delay,
                    )
                    time.sleep(delay)
            raise last_exc  # should be unreachable
        return wrapper
    return decorator


def fixed_delay_retry(
    retries: int = 3,
    delay: int = 30,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry with a constant delay (used for RotoWire to avoid triggering rate-limits)."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == retries:
                        logger.error(
                            "%s failed after %d attempt(s): %s",
                            func.__qualname__, retries + 1, exc,
                        )
                        raise
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %ds",
                        func.__qualname__, attempt + 1, retries, exc, delay,
                    )
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
