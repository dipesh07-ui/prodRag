import logfire

def parse_text(file_path:str):
    """
    Parse a text file and return the clean text
    """
    with logfire.span("📄 TEXT PARSING",filename=file_path):
        try:
            with open(file_path,'r',encoding='utf-8',errors='ignore') as f:
                content = f.read()

        except Exception as e:
            logfire.error(f"❌ Text Parse Failed: {e}")

            raise e
