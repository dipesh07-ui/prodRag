from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logfire


def chunk_text(text:str, chunk_size:int=1500) ->List[str]:
    """
    Simple semantic chunker that splits text into chunks of a specified size (default 1500 characters).
    """

    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_text(text)
        logfire.info(f"✅ Generated {len(chunks)} chunks")

        return chunks

