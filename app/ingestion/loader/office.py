import logfire
from unstructured.partition.auto import partition

def parse_office(file_path:str):
    """"
    Parse an office document (docx, pptx, xlsx) using unstructured and return the clean text
    Unlike pdf, these formats are unstructured and lightweight, so we can parse locally
    """

    with logfire.span("📄 OFFICE PARSING",filename=file_path):
        try:
            elements = partition(file_path)
            full_text = "\n".join([str(el) for el in elements])

            if not full_text.strip():
                logfire.warning(f"⚠️ Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"✅ Successfully parsed {len(full_text)} characters")

            return full_text

        except Exception as e:
            logfire.error(f"❌ Office Parse Failed: {e}")
            raise e
