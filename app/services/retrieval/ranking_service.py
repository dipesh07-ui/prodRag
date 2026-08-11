import time
import logfire
from flashrank import Ranker, RerankRequest

_ranker=None

def _get_ranker() -> Ranker:
    """
    Initialize and return the FlashRank Ranker instance. If already initialized, return the existing instance.
    It use local onnx model (TinyBERT) for ultra-fast re-ranking of search results.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            _ranker = Ranker(cache_dir="/temp/flashrank")
            logfire.info("🧠 FlashRank Model (TinyBERT) initialized successfully.")
        except Exception:
            _ranker = Ranker()
    return _ranker



def rerank_documents(query:str, documents:list[dict], top_n:int=5) -> list[dict]:
    """ Refines retrieval results by re-scoring documents against the query semantically.
    
    Why FlashRank? 
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []
    
    start_time = time.time()
    logfire.info(f"📡 Re-ranking {len(documents)} documents ...")

    try:
        ranker = _get_ranker()
        passages = [
            {"id":i , "text":doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'

        logfire.info(f"✅ Re-ranking completed in {duration:.2f} seconds. Top score: {top_score}")
        return reranked_docs
    
    except Exception as e:
        logfire.error(f"❌ Re-ranking failed: {e}")
        return documents[:top_n] # Return original documents if re-ranking fails