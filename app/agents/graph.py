from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.responder import generate_response
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.planner import planner_node

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_response)


def route_planner(state:AgentState):

    if state["current_query"] == "CONVERSATIONAL":
        return "responder"

    return "retriever"


workflow.set_conditional_entry_point("planner")

workflow.add_conditional_edges(
        "planner",
        route_planner,
        {"retriever": "retriever", "responder": "responder"}
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)

# --- MEMORY UPGRADE ---
# MemorySaver allows the agent to remember conversations based on 'thread_id'
checkpointer = MemorySaver()


rag_agent = workflow.compile(checkpointer=checkpointer)