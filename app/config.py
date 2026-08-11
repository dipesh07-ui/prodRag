import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # -- GEMINI EMBEDDINGS
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --  QDRANT VECTOR DATABASE
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # -- GROQ REASONING MODEL
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "groqrag"     # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "groqrag1"  # fallback: @brag/llama-3.1-8b-instant


settings = Settings()

