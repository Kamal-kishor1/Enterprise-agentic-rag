import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


async def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    Preserving source context for claim-level citations.
    """
    query = state["current_query"]

    # Standard Retrieval Logic
    with logfire.span(" Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")

        # 1. Fetch raw candidate points (returns list of dicts: {"content", "source", "score"})

        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        if not raw_results:
            logfire.warning("Vector DB returned 0 documents.")
            return {
                "documents": [],
                "status": "No technical context found.",
                "plan": state["plan"] + ["Context Retrieved: Empty"],
            }

        # 2. Extract content list for FlashRank cross-encoding
        doc_contents = [doc["content"] for doc in raw_results]

        with logfire.span(" Semantic Reranking"):
            # FlashRank returns the top 5 reranked content strings
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

        # 3. Map reranked contents back to their original source metadata
        content_to_source = {
            doc["content"]: doc.get("source", "Unknown Document") for doc in raw_results
        }

        # 4. Format each chunk with structured XML tags containing the source filename
        formatted_docs = []
        for i, text in enumerate(reranked_contents, 1):
            source_file = content_to_source.get(text, "Unknown Document")
            formatted_chunk = (
                f"<document id='{i}' source='{source_file}'>\n{text}\n</document>"
            )
            formatted_docs.append(formatted_chunk)

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
