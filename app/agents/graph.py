from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node

# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)


# 3. Define the Edges and Routing Logic
def route_planner(state: AgentState):
    # Routes the workflow based on the planner's decision

    if state["current_query"] == "CONVERSATIONAL":
        return "responder"

    return "retriever"


# Workflow start from this planner node
workflow.set_entry_point("planner")


# Conditional Edges: Planner -> Router -> (Retriever or Responder)
workflow.add_conditional_edges(
    "planner", route_planner, {"retriever": "retriever", "responder": "responder"}
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# ---- Memory Upgrade ----
# MemorySaver allows the agent to remember conversation based on the memory
#  4. Production Checkpointing (PostgreSQL with Safe Fallback)
def init_checkpointer():
    """
    Initializes durable PostgreSQL checkpointing if DATABASE_URL is configured.
    Falls back to in-memory MemorySaver if no database is provided.
    """
    db_url = settings.DATABASE_URL
    if db_url:
        try:
            from psycopg_pool import ConnectionPool
            from langgraph.checkpoint.postgres import PostgresSaver

            # Pool manages concurrent worker connections efficiently
            pool = ConnectionPool(
                conninfo=db_url, max_size=20, kwargs={"autocommit": True}
            )
            postgres_saver = PostgresSaver(pool)
            # setup() runs database migrations / creates tables automatically!
            postgres_saver.setup()
            print(" Connected to PostgreSQL: Persistent checkpointer active.")
            return postgres_saver
        except Exception as e:
            print(f" PostgreSQL connection failed: {e}. Falling back to MemorySaver.")
            return MemorySaver()
    print(" No DATABASE_URL found. Using in-memory MemorySaver (single-worker only).")
    return MemorySaver()


checkpointer = init_checkpointer()

# 5. Compile the Graph with Durable Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
