import asyncio
from aiolimiter import AsyncLimiter
from src.utils.config import get_rate_limiter_config

# Singleton instance
_rate_limiter = None

def get_rate_limiter():
    """Returns the global rate limiter instance (singleton)."""
    global _rate_limiter
    if _rate_limiter is None:
        max_rate, time_period = get_rate_limiter_config()
        _rate_limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
    return _rate_limiter