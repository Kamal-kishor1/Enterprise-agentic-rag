from app.agents.state import AgentState
from app.services.llm_gateway import gateway_llm

import logfire

# Direct groq call - the llm Gateway (litellm)
# Planner uses non-streaming gateway for fast internal intent classification
llm = gateway_llm


async def planner_node(state: AgentState):
    # the planner determines if a search is needed based on the entire conversation.

    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.


    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    {user_message}

    Task:
    1. If the latest message is a greeting (hi, hello) or conversational question that can be
    answered using ONLY the conversation history above, respond "CONVERSATIONAL".
    2. If it is a technical question requiring fresh documentation,
    output a clean semantic search query (3 to 6 key terms).
    DO NOT output web search operators (no "site:", no URLs, no Boolean operators).

    Output ONLY "CONVERSATIONAL" or the clean search query string.
    """

    with logfire.span(" Planner Decision"):
        res = await llm.ainvoke(prompt)
        decision = res.content.strip()
        logfire.info(f"Intent identified: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Converstional/Memory", "Retrieval: Skipped"],
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: TEchnical", f"Search Term : {decision}"],
    }
