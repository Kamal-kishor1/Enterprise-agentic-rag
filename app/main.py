from concurrent.futures import thread

import json
from unittest import result
from fastapi.responses import StreamingResponse

import logfire
import os
from dotenv import load_dotenv
from .config import settings

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import rag_agent
from app.services.cache_service import get_cached_query, set_cached_query
from app.services.guardrails import (
    sanitize_pii,
    check_prompt_injection,
    sanitize_output,
)

# Initialize FastAPI
app = FastAPI(
    title="Agentic RAG API",
    version="0.0.1",
    description="THIS IS THE K8S AGENTIC RAG SYSTEM. ",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Use the allowed origins from settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = (
        "default_user"  # in langgraph session id is created as thread id
    )


@app.get("/")
def home():
    return {"message": " LangGraph RAG API is live. "}


@app.get("/graph")
def get_graph_image():
    """
    Retures the graph image of the agent's workflow.
    """

    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")

    except Exception as e:
        return {"error": f" Could not generate graph image: {e}"}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a Post request.
    """

    # 1. PII Sanitization
    clean_query, pii_found = sanitize_pii(request.q)

    # 2. Prompt Injection Guardrail (Blocks attacks in <120ms without invoking RAG)
    is_attack, score = check_prompt_injection(clean_query)
    if is_attack:
        return {
            "question": request.q,
            "answer": "Security Alert: This request was blocked because it triggered prompt injection and safety guardrails.",
            "thought_process": ["Security policy violation detected."],
            "status": "blocked",
            "sources": [],
            "cached": False,
        }

    q = clean_query  # query
    thread_id = request.thread_id  # session id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }

    # 1. Check Cache First
    cached_response = await get_cached_query(q)
    if cached_response:
        cached_response["cached"] = True
        return cached_response

    # 2. Configuration for Memory (Thread id)
    config = {"configurable": {"thread_id": thread_id}}
    # new thread id and continuously creating new thread id

    try:
        final_output = await rag_agent.ainvoke(initial_state, config=config)

        result = {
            "question": q,
            "answer": sanitize_output(final_output.get("final_answer") or ""),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "cached": False,
        }

        # 3. Store verified answer in cache asynchronously for future user
        await set_cached_query(q, result)

        return result

    except Exception as e:
        logfire.error(f" Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologies, but I encountered an internal server error while processing",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streams LLM tokens in real-time as Server-Sent Events (SSE).
    """
    # 1. PII Sanitization
    clean_query, _ = sanitize_pii(request.q)
    # 2. Prompt Injection Guardrail
    is_attack, score = check_prompt_injection(clean_query)
    if is_attack:

        async def block_stream():
            yield "Security Alert: This request was blocked by enterprise security guardrails."

        return StreamingResponse(block_stream(), media_type="text/plain")

    q = clean_query  # query
    thread_id = request.thread_id  # session id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }

    # 1. Check Cache for instant stream emission
    cached_response = await get_cached_query(q)
    if cached_response:
        # Stream the cached answer as a single event
        async def cached_stream():
            yield cached_response.get("answer", "")

        return StreamingResponse(cached_stream(), media_type="text/plain")

    # 2. Configuration for Memory (Thread id)
    config = {"configurable": {"thread_id": thread_id}}
    # new thread id and continuously creating new thread id

    async def token_generator():
        # Initialize accumulator list
        full_response = []

        # 3. astream_events v2 captures real-time token chunks from the LLM
        async for event in rag_agent.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event["event"]
            # When the Responder's ChatGroq emits a new token chunk
            if kind == "on_chat_model_stream":
                # Ensure the token comes from the responder node, not planner
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name == "responder":
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content:
                        # Collect each chunk
                        full_response.append(chunk_content)

                        # Send text chunk directly to the stream
                        yield chunk_content

        # Store streamed response in cache once generation completes
        if full_response:
            await set_cached_query(
                q, {"question": q, "answer": "".join(full_response), "sources": []}
            )

    # media_type="text/plain" or "text/event-stream" keeps the HTTP pipe open
    return StreamingResponse(token_generator(), media_type="text/plain")
