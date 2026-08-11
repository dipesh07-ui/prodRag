import logfire 
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig , LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLOANG_CONTENT , YAML_CONTENT , RAIL_INDICATORS

_rails : LLMRails | None = None

def initialize_rails() -> None:
    """
    Initialize the nemoguardrails singletion on start , use llama-3.1-8b-instant for intent classification and heavier llama-3.3-70b-versatile is reserved for rag pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key = settings.GROQ_API_KEY,
        model = "llama-3.1-8b-instant",
        temperature = 0
    )

    config = RailsConfig.from_content(
        colang_content = COLOANG_CONTENT,
        yaml_content = YAML_CONTENT
    )

    _rails = LLMRails(config , llm = guard_llm )

    logfire.info("🛡️ NeMo Guardrails initialised (llama-3.1-8b-instant).")

def guard(message:str)-> tuple[bool , str |None]:
    """
    runs user query through nemo railsgate ..

    Returns :
        (True, rail_response) rail fired skip the rag pipeline entirely 
        (False, None) rail not fired , continue with rag pipeline
    """

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False , None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(message=[{"role":"user","content":message}])

        content = result.get("content" , "") if isinstance(result, dict) else str(result)
        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query = '{message[:80]}' ")
            return True , content


        logfire.info("✅ Guardrails Passed")
        return False , None
    