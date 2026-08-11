from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query
import logfire


client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key = settings.QDRANT_API_KEY
)


def search_enterprise_knowledge(query:str, limit:int=8):
    """Search for relevant enterprise knowledge based on the query in Qdrant database and return the results."""
    try:
        # embed the query to get its vector representation
        query_vector = embed_query(query)

        # using query_points , the modern way to search in Qdrant
        response = client.query_points(
            collection_name = settings.QDRANT_COLLECTION_NAME,
            query_vector = query_vector,
            limit = limit,
            with_payload = True 
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", ""),
                "score": res.score
            })

        return results
    
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []