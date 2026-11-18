import json
import os
import time
import logfire
from qdrant_client import QdrantClient
from tenacity import retry, stop_after_attempt, wait_fixed

from app.config import settings
from app.services.retrieval.embeddings import embed_query

# 1. Initialize Qdrant Client with 10.0s connection timeout for AWS cloud latency
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=10.0,  # Accommodates transatlantic TLS handshake
)

# 2. In-Memory Circuit Breaker State
_consecutive_failures = 0
_circuit_open_until = 0.0
CIRCUIT_FAILURE_THRESHOLD = (
    1  # Fail-Fast: Trip circuit immediately on first connection failure
)
CIRCUIT_RECOVERY_TIME = 60.0  # Stay open for 60s before testing cloud again


def _is_circuit_open() -> bool:
    """Returns True if the circuit breaker is currently open."""
    global _circuit_open_until
    return _circuit_open_until > time.time()


def _record_success():
    """Reset failure counter when cloud search succeeds."""
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures = 0
    _circuit_open_until = 0.0


def _record_failure():
    """Increment failure counter and trip circuit if threshold exceeded."""
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures += 1
    if _consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
        _circuit_open_until = time.time() + CIRCUIT_RECOVERY_TIME
        logfire.error(
            f" Qdrant Circuit Breaker TRIPPED! Suppressing cloud calls for {CIRCUIT_RECOVERY_TIME}s."
        )


def _search_local_fallback(limit: int = 15) -> list[dict]:
    """
    Emergency Local Fallback: If Qdrant Cloud drops connection,
    read raw chunks from processed_data/ (prioritizing ground-truth data).
    """
    fallback_docs = []
    dirs_to_check = [
        os.path.join("processed_data", "true"),
        os.path.join("processed_data", "noisy"),
    ]

    for p_dir in dirs_to_check:
        if os.path.exists(p_dir):
            for root, _, files in os.walk(p_dir):
                for f in files:
                    if f.endswith(".json"):
                        try:
                            with open(os.path.join(root, f), "r", encoding="utf-8") as fp:
                                data = json.load(fp)
                                filename = data.get("filename", f)
                                for chunk in data.get("chunks", []):
                                    fallback_docs.append(
                                        {
                                            "content": chunk,
                                            "source": filename,
                                            "score": 0.5,
                                        }
                                    )
                        except Exception:
                            continue

    return fallback_docs[: limit * 2]


# 3. Fast Retry: only 1 attempt with a fast 0.5s pause
@retry(
    stop=stop_after_attempt(1),
    # wait=wait_fixed(0.5),
    reraise=True,
)
def _search_qdrant_direct(query: str, limit: int = 15):
    """Direct cloud search with strict timeout and fast 0.5s retry."""
    query_vector = embed_query(query)

    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    results = []
    for res in response.points:
        results.append(
            {
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score,
            }
        )
    return results


def search_enterprise_knowledge(query: str, limit: int = 15):
    """
    Public Retrieval Entrypoint protected by Circuit Breaker & Fail-Fast Fallback.
    """
    # Check circuit status
    if _is_circuit_open():
        remaining = int(_circuit_open_until - time.time())
        logfire.warning(
            f"Qdrant circuit is OPEN ({remaining}s remaining). Routing to local fallback."
        )
        return _search_local_fallback(limit=limit)

    try:
        results = _search_qdrant_direct(query, limit=limit)
        _record_success()
        return results

    except Exception as e:
        logfire.error(f"Qdrant Search Failed: {e}")
        _record_failure()
        logfire.info("Engaging emergency local fallback.")
        return _search_local_fallback(limit=limit)
