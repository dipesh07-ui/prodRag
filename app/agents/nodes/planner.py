from app.agents.state import AgentState
from langchain_groq import ChatGroq
from app.config import settings
import logfire


llm = ChatGroq(
    api_key = settings.GROQ_API_KEY,
    model = settings.GROQ_MODEL,
    fallback_api_key = settings.GROQ_FALLBACK_API_KEY,
    temperature = 0.4
    )

def planner_node(state:AgentState):
    """"
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """

    history = "" 
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    
    user_query = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_query}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent Identified : {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "HANDLING CONVERSATION USING HISTORY",
            "plan": ["INTENT CONVERSATIONAL/MEMORY", "RETRIVAL NOT NEEDED"],
        }
    
    
    return {
        "current_query": decision,
        "status": f"TECHNICAL SEARCH NEEDED, SEARCHING FOR {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"],
    }
    