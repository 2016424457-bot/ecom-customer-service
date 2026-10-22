import asyncio
from pathlib import Path
from typing import Any, Callable, TypeVar

from pymilvus import DataType, MilvusClient

from app.config import Settings

T = TypeVar("T")

_CLIENTS: dict[str, MilvusClient] = {}
_ENSURED: set[tuple[int, str, int]] = set()


def reset_client_cache() -> None:
    for client in _CLIENTS.values():
        close = getattr(client, "close", None)
        if close is not None:
            close()
    _CLIENTS.clear()
    _ENSURED.clear()


def get_client(settings: Settings) -> MilvusClient:
    uri = settings.milvus_uri
    client = _CLIENTS.get(uri)
    if client is None:
        if not uri.startswith("http"):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(uri=uri)
        _CLIENTS[uri] = client
    return client


async def acall(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(fn, *args, **kwargs)


def ensure_collection(
    client: Any, collection: str, dim: int = 1024
) -> None:
    cache_key = (id(client), collection, dim)
    if cache_key in _ENSURED:
        return
    if client.has_collection(collection):
        client.load_collection(collection)
        _ENSURED.add(cache_key)
        return

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(collection)
    _ENSURED.add(cache_key)


def upsert_vectors(
    client: Any,
    ids: list[int],
    vectors: list[list[float]],
    collection: str,
) -> None:
    if not ids:
        return
    client.upsert(
        collection_name=collection,
        data=[
            {"id": int(chunk_id), "dense": list(vector)}
            for chunk_id, vector in zip(ids, vectors, strict=True)
        ],
    )


def dense_search(
    client: Any,
    vector: list[float],
    top_k: int,
    collection: str,
) -> list[tuple[int, float]]:
    results = client.search(
        collection_name=collection,
        data=[list(vector)],
        limit=top_k,
        output_fields=["id"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        (int(hit["id"]), float(hit["distance"])) for hit in results[0]
    ]


def delete_vectors(
    client: Any, ids: list[int], collection: str
) -> int:
    if not ids:
        return 0
    result = client.delete(
        collection_name=collection,
        ids=[int(chunk_id) for chunk_id in ids],
    )
    if isinstance(result, dict):
        return int(result.get("delete_count", len(ids)))
    return len(ids)


def has_ids(
    client: Any, ids: list[int], collection: str
) -> set[int]:
    if not ids:
        return set()
    rows = client.get(
        collection_name=collection,
        ids=[int(chunk_id) for chunk_id in ids],
        output_fields=["id"],
    )
    return {int(row["id"]) for row in rows}


def count_vectors(client: Any, collection: str) -> int:
    stats = client.get_collection_stats(collection)
    return int(stats.get("row_count", 0))
