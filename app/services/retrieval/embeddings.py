import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding

# BM25 Sparse Embedding Model Singleton
_bm25_model = None

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2


_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# Initialize the BM25 Model Singleton
def get_bm25_model():
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


# Add the helper function embed_sparse(texts: list[str])
def embed_sparse(texts: list[str]):
    """Generate sparse BM25 representations (indices & Values) for chunks."""
    model = get_bm25_model()

    return list(model.embed(texts))


# Add a query helper function
def embed_sparse_query(query: str):
    """Generate a single sparse BM25 vector for search queries."""
    model = get_bm25_model()
    return list(model.embed([query]))[0]


def _probe_gemini():
    # Try one embed call to verify Gemini is reachable. Returns model or None.
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probs")
        logfire.info(" Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model

    except Exception as e:
        logfire.warning(
            f" Gemini probe failded: {e}. Will use sentence-transformers fallback."
        )
        return None


def _load_fallback():
    try:
        from sentence_transformers import SentenceTransformer

        logfire.info(
            " Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim)."
        )
        return SentenceTransformer("all-mpnet-base-v2")
    except ImportError:
        logfire.warning(
            " sentence-transformers not installed; running in Gemini-only cloud mode."
        )
        return None


def _init():
    global _active_model, _model_type

    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"

    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


def get_embedding_dim() -> int:
    # Return the vector dimension for the active model. Call after _init().

    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


def _embed_batch(batch: list[str]) -> list[list[float]]:

    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)

            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(
                    x in err for x in ("429", "rate", "quota", "resource_exhausted")
                )

                if is_rate_limit and attempt < 3:
                    wait = 2**attempt
                    logfire.warning(
                        f" Gemini rate limit hit - retrying in {wait}"
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)

                else:
                    logfire.error(f" Gemini embedding failed: {e}")
                    raise

        raise RuntimeError(" Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:

    _init()

    if _model_type == "gemini":
        return _active_model.embed_query(query)

    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:

    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))

    return all_embeddings
