import logfire
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from app.agents.state import AgentState

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """

    query = state["current_query"]

    with logfire.span("🔍 Knowledge Search"):
        logfire.info(f"Searching for relevant knowledge for query: '{query}'")
        documents = search_enterprise_knowledge(query,limit=15)
        logfire.info(f"Found {len(documents)} documents from Qdrant.")

        doc_contents = [doc['content'] for doc in documents]

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_docs = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking completed. Kept The Top 5 documents .")

            formatted_docs = [f"CONTENT {doc}" for doc in reranked_docs]


    return {
        "documents": formatted_docs,
        "status": "Found Technical Knowledge",
        "plan": state["plan"] + ["Context Retrieved"]
    }



