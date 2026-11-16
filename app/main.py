from concurrent.futures import thread

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
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a Post request.
    """

    q = request.q  # query
    thread_id = request.thread_id  # session id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }

    # Configuration for Memory (Thread id)
    config = {"configurable": {"thread_id": thread_id}}
    # new thread id and continuously creating new thread id

    try:
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
        }

    except Exception as e:
        logfire.error(f" Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologies, but I encountered an internal server error while processing",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }
