from app.config import Settings
from app.knowledge import milvus_client


def test_empty_collection_search_returns_empty(tmp_path, clean_unit_env):
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        milvus_uri=str(tmp_path / "empty.db"),
    )
    milvus_client.reset_client_cache()
    client = milvus_client.get_client(settings)
    milvus_client.ensure_collection(client, "empty")

    assert (
        milvus_client.dense_search(
            client, [0.0] * 1024, 3, "empty"
        )
        == []
    )
    milvus_client.reset_client_cache()


def test_milvus_lite_round_trip(tmp_path, clean_unit_env):
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        milvus_uri=str(tmp_path / "milvus.db"),
    )
    milvus_client.reset_client_cache()
    client = milvus_client.get_client(settings)
    collection = "knowledge"
    milvus_client.ensure_collection(client, collection)

    first = [1.0] + [0.0] * 1023
    second = [0.0, 1.0] + [0.0] * 1022
    milvus_client.upsert_vectors(client, [1, 2], [first, second], collection)

    hits = milvus_client.dense_search(client, first, 2, collection)
    assert hits[0][0] == 1
    assert milvus_client.has_ids(client, [1, 2], collection) == {1, 2}

    assert milvus_client.delete_vectors(client, [1], collection) == 1
    assert milvus_client.has_ids(client, [1], collection) == set()
    milvus_client.reset_client_cache()
