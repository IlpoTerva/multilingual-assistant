from __future__ import annotations
from typing import Any, Annotated, TypedDict
import json
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a helpful assistant that can answer questions and perform tasks using the tools provided.

"""

def _make_llm(with_tools:bool)-> Any:
    """Create a LangChain LLM instance with or without tools."""
    llm = ChatOpenAI(
        base_url=settings.OLLAMA_BASE_URL,
        api_key="ollama",
        model_name=settings.OLLAMA_MODEL,
        temperature=0,
    )

    llm_fallback = ChatOpenAI(
        base_url=settings.OLLAMA_BASE_URL,
        api_key="ollama",
        model_name=settings.OLLAMA_MODEL_FALLBACK,
        temperature=0,
    )

    if with_tools:
        llm = llm.bind_tools(TOOL_SCHEMAS)
        llm_fallback = llm_fallback.bind_tools(TOOL_SCHEMAS)
    return llm.with_fallbacks([llm_fallback])
