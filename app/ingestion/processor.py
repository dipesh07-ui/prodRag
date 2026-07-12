import os
import sys
import logfire
import uuid
import json


from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import get_embedding_dim , embed_texts
from app.ingestion.loader.pdf import parse_pdf
from app.ingestion.loader.txt import parse_text
from app.ingestion.loader.docx import parse_office
from app.ingestion.loader.html import parse_html

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"


qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key = settings.QDRANT_API_KEY
)




