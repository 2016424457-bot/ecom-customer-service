from types import SimpleNamespace

import pytest

from app.config import Settings
from app.knowledge import embeddings


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test",
        embed_base_url="https://embed.test/v1",
        embed_api_key="embed-key",
    )


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def create(self, *, model: str, input: list[str]):
        self.calls.append((model, input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index)] * 1024, index=index)
                for index in range(len(input))
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsResource()


async def test_embed_texts_uses_model_and_preserves_order(
    settings: Settings,
) -> None:
    embeddings.reset_client_cache()
    fake = FakeClient()
    embeddings._CLIENTS[("https://embed.test/v1", "embed-key")] = fake

    vectors = await embeddings.embed_texts(["第一段", "第二段"], settings)

    assert fake.embeddings.calls == [("BAAI/bge-m3", ["第一段", "第二段"])]
    assert vectors == [[0.0] * 1024, [1.0] * 1024]
    embeddings.reset_client_cache()


async def test_embed_query_returns_first_vector(settings: Settings) -> None:
    embeddings.reset_client_cache()
    fake = FakeClient()
    embeddings._CLIENTS[("https://embed.test/v1", "embed-key")] = fake

    vector = await embeddings.embed_query("邮费是多少", settings)

    assert vector == [0.0] * 1024
    assert fake.embeddings.calls == [("BAAI/bge-m3", ["邮费是多少"])]
    embeddings.reset_client_cache()


async def test_embed_texts_empty_input_skips_api(settings: Settings) -> None:
    embeddings.reset_client_cache()
    fake = FakeClient()
    embeddings._CLIENTS[("https://embed.test/v1", "embed-key")] = fake

    assert await embeddings.embed_texts([], settings) == []
    assert fake.embeddings.calls == []
    embeddings.reset_client_cache()


async def test_embed_texts_requires_api_key(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        embed_api_key="",
    )

    with pytest.raises(RuntimeError, match="EMBED_API_KEY"):
        await embeddings.embed_texts(["x"], settings)


async def test_embed_texts_sorts_by_response_index(
    settings: Settings,
) -> None:
    embeddings.reset_client_cache()

    class ShuffledResource(FakeEmbeddingsResource):
        async def create(self, *, model, input):
            self.calls.append((model, input))
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[1.0] * 1024, index=1),
                    SimpleNamespace(embedding=[0.0] * 1024, index=0),
                ]
            )

    fake = FakeClient()
    fake.embeddings = ShuffledResource()
    embeddings._CLIENTS[("https://embed.test/v1", "embed-key")] = fake

    vectors = await embeddings.embed_texts(["第一段", "第二段"], settings)

    assert vectors == [[0.0] * 1024, [1.0] * 1024]
    embeddings.reset_client_cache()


async def test_embed_texts_rejects_duplicate_indices(
    settings: Settings,
) -> None:
    embeddings.reset_client_cache()

    class DuplicateResource(FakeEmbeddingsResource):
        async def create(self, *, model, input):
            self.calls.append((model, input))
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[0.0] * 1024, index=0),
                    SimpleNamespace(embedding=[1.0] * 1024, index=0),
                ]
            )

    fake = FakeClient()
    fake.embeddings = DuplicateResource()
    embeddings._CLIENTS[("https://embed.test/v1", "embed-key")] = fake

    with pytest.raises(RuntimeError, match="indices"):
        await embeddings.embed_texts(["a", "b"], settings)
    embeddings.reset_client_cache()
