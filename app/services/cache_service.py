import hashlib
import json
import time
from typing import Any, Optional, Dict
import logfire

from app.config import settings

# 1. In-memory cache Fallback Structure: {hash: {"data": dict, "expires_at": float}}
_memory_cache: Dict[str, Dict[str, Any]] = {}
_redis_client = None


async def _get_redis():
    """Lazily initialize Redis client if REDIS_URL is configured."""
    global _redis_client
    if settings.REDIS_URL and _redis_client is None:
        try:
            import redis.asyncio as aioredis

            _redis_client = aioredis.from_url(
                settings.REDIS_URL, decode_responses=True, socket_timeout=2 - 0
            )
            # Test connection
            await _redis_client.ping()
            logfire.info("Connected to Redis: Enterprise response cache active.")
        except Exception as e:
            logfire.warning(
                f"Redis connection failed: {e}. Falling back to In-Memory cache."
            )
            _redis_client = None
    return _redis_client


def _hash_query(query: str) -> str:
    """Normalize query text and generate a deterministic SHA-256 cache key."""
    # Strip whitespace, lowercase, and remove trailing punctuation
    normalized = query.strip().lower().rstrip("?.!,:;")
    return f"rag:cache:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


async def get_cached_query(query: str) -> Optional[dict]:
    """
    Checks cache for a previously verified answer.
    Returns the answer dictionary if found and not expired, else None.
    """
    cache_key = _hash_query(query)
    redis_client = await _get_redis()

    if redis_client:
        try:
            raw = await redis_client.get(cache_key)
            if raw:
                logfire.info(f"Redis Cache hit for query: '{query[:40]}...'")
                return json.loads(raw)
        except Exception as e:
            logfire.warning(f"Redis get failed: {e}. Falling back to In-Memory cache.")

    # Fallback to in-memory cache
    entry = _memory_cache.get(cache_key)
    if entry:
        if entry["expires_at"] > time.time():
            logfire.info(f"In-Memory Cache hit for query: '{query[:40]}...'")
            return entry["data"]
        else:
            # Expired entry
            del _memory_cache[cache_key]  # clean up expired entry

    return None


async def set_cached_query(query: str, data: dict, ttl: Optional[int] = None):
    """
    Stores a synthesized answer in cache with a configurable TTL.
    """
    cache_key = _hash_query(query)
    ttl = ttl or settings.CACHE_TTL_SECONDS
    redis_client = await _get_redis()
    if redis_client:
        try:
            await redis_client.set(cache_key, json.dumps(data), ex=ttl)
            logfire.info(f"Stored query in Redis cache with TTL {ttl}s.")
            return
        except Exception as e:
            logfire.warning(f"Redis set failed: {e}")
    # Fallback to In-Memory cache
    _memory_cache[cache_key] = {"data": data, "expires_at": time.time() + ttl}
    logfire.info(f"Stored query in In-Memory cache with TTL {ttl}s.")
