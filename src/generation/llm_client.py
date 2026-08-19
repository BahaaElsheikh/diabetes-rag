"""
LLM Client Wrapper for Day 3 (Grounded Generation).

Provides a provider-agnostic interface for invoking the underlying LLM.
Uses Google Gemini (google-genai SDK) with structured JSON output support.
"""

from __future__ import annotations

import os
import warnings
from typing import Type
from google import genai
from google.genai import types
from pydantic import BaseModel

# Suppress harmless pydantic internal type warnings from google.genai
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL = "gemini-2.5-flash-lite"

_client: genai.Client | None = None


def get_llm_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set!")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_response(
    system_prompt: str,
    user_prompt: str,
    response_schema: Type[BaseModel] | None = None,
    temperature: float = 0.0,
    max_retries: int = 1,
) -> str:
    """Generate a text response from the LLM.

    Args:
        system_prompt: High-level instructions (grounding rules, constraints).
        user_prompt: User question and retrieved context.
        response_schema: Optional Pydantic model for structured JSON output.
        temperature: Sampling temperature (default 0.0 for strict deterministic Q&A).
        max_retries: Number of retries on 429/quota error.

    Returns:
        String output (JSON formatted if response_schema is provided).
    """
    import time
    client = get_llm_client()

    config_args = {
        "system_instruction": system_prompt,
        "temperature": temperature,
    }

    if response_schema is not None:
        config_args["response_mime_type"] = "application/json"
        config_args["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_args)

    candidate_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

    for attempt in range(max_retries):
        for model in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=config,
                )
                return response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota" in err_str:
                    continue  # Try next fallback model
                raise e

        # If all models hit 429, wait briefly before retrying loop
        time.sleep((attempt + 1) * 3.0)

    raise RuntimeError("All candidate LLM models exhausted rate limits.")
