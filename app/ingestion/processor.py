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
from app.ingestion.loader.text import parse_text
from app.ingestion.loader.office import parse_office
from app.ingestion.loader.html import parse_html
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"


qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key = settings.QDRANT_API_KEY
)


def save_processed_locally(data:dict,filename:str,source_type:str):
    """save processed data locally as json file"""
    os.makedirs(PROCESSED_DATA_DIR,exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_DIR,f"{filename}.json")
    with open(file_path,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    return file_path




def process_file(file_path:str,filename:str,source_type:str):
    """parse -> chunks -> saveLocally -> embed -> index in Qdrantdb"""

    with logfire.span("Processing file", file=filename, source=source_type):
        try:
            ext = filename.lower().rsplit(".",1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html","htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx","pptx"):
                full_text = parse_office(file_path)
            
            else :
                logfire.warning(f"skipping unsupported file {filename}")
                return 
            
            if not full_text or not full_text.strip():
                logfire.warning(f"no text extracted from {filename}")
                return
            
            # chunking
            chunks = chunk_text(full_text)
            if not chunks:
                return
            
            # save locally 
            processed_data = {
                "filename" : filename ,
                "source_type" :source_type ,
                "chunks" : chunks
            }

            local_path = save_processed_locally(processed_data,filename=filename,source_type=source_type)
            logfire.info(f"✅ Processed data saved locally at -> {local_path}")

            # embed and index in Qdrant

            with logfire.span("Vectorizing & Indexing"):
                embedding = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id = str(uuid.uuid4()),
                        vector = vector,
                        payload = {
                            "text" : chunk,
                            "source" : filename,
                            "source_type" : source_type
                        },
                    )
                    for chunk, vector in zip(chunks, embedding)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )

                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
        
        except Exception as e:
            logfire.error(f"Error processing file {filename}: {str(e)}")


def process_directory(dir_path:str,source_type:str):
    """process all files in a directory"""
    with logfire.span("Processing directory", directory=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")

        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir:str,explicit_source_type:str =None , wipe:bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal ingestion started", base_dir=base_dir):
        if wipe:

            with logfire.span("Wiping Qdrant collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Deleted existing Qdrant collection: {settings.QDRANT_COLLECTION}")
         
        
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.recreate_collection(
                collection_name = settings.QDRANT_COLLECTION,
                vectors_config = models.VectorParams(
                    size = dim,
                    distance = models.Distance.COSINE
                ),
            )

            logfire.info(f"Created new Qdrant collection: {settings.QDRANT_COLLECTION} with dimension {dim}")


        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")
