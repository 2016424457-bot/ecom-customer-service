from openai import AsyncOpenAI

from app.config import Settings

_CLIENTS: dict[tuple[str, str], AsyncOpenAI] = {}


def _get_client(settings: Settings) -> AsyncOpenAI:
    """Reuse one AsyncOpenAI client per (base_url, api_key)."""
    key = (settings.embed_base_url, settings.embed_api_key)
    client = _CLIENTS.get(key)
    if client is None:
        client = AsyncOpenAI(
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
        )
        _CLIENTS[key] = client
    return client


def reset_client_cache() -> None:
    _CLIENTS.clear()


async def embed_texts(
    texts: list[str], settings: Settings
) -> list[list[float]]:
    if not texts:
        return []
    if not settings.embed_api_key:
        raise RuntimeError(
            "EMBED_API_KEY is not set; cannot call the embedding API"
        )
    response = await _get_client(settings).embeddings.create(
        model=settings.embed_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def embed_query(text: str, settings: Settings) -> list[float]:
    vectors = await embed_texts([text], settings)
    return vectors[0]
