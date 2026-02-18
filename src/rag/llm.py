"""Ollama LLM client via OpenAI-compatible API."""

from collections.abc import Generator

from openai import OpenAI

from rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return a singleton OpenAI client pointing to Ollama."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't need a real key
        )
    return _client


def generate(prompt: str, temperature: float = 0.1) -> str:
    """Generate a complete response from the LLM."""
    client = get_client()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_stream(prompt: str, temperature: float = 0.1) -> Generator[str, None, None]:
    """Stream response tokens from the LLM."""
    client = get_client()
    stream = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
