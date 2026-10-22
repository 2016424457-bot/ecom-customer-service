import asyncio
from pathlib import Path
from typing import Any, Callable, TypeVar

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import Settings
from app.knowledge.types import VectorRecord

T = TypeVar("T")

_CLIENTS: dict[str, MilvusClient] = {}
_ENSURED: set[tuple] = set()

REQUIRED_FIELDS = frozenset(
    {
        "id",
        "text",
        "sparse",
        "dense",
        "category",
        "content_type",
        "section_path",
        "source_doc",
    }
)


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
    client: Any,
    collection: str,
    dim: int = 1024,
    analyzer_type: str = "jieba",
    max_length: int = 4096,
) -> None:
    cache_key = (id(client), collection, dim, analyzer_type)
    if cache_key in _ENSURED:
        return
    if client.has_collection(collection):
        description = client.describe_collection(collection)
        fields = {
            field["name"] for field in description.get("fields", [])
        }
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise RuntimeError(
                "Milvus collection schema is outdated; run "
                "build_knowledge.py --recreate"
            )
        client.load_collection(collection)
        _ENSURED.add(cache_key)
        return

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=max_length,
        enable_analyzer=True,
        analyzer_params={"type": analyzer_type},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("category", DataType.VARCHAR, max_length=255)
    schema.add_field("content_type", DataType.VARCHAR, max_length=32)
    schema.add_field("section_path", DataType.VARCHAR, max_length=512)
    schema.add_field("source_doc", DataType.VARCHAR, max_length=255)
    schema.add_function(
        Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(collection)
    _ENSURED.add(cache_key)


def recreate_collection(
    client: Any,
    collection: str,
    dim: int = 1024,
    analyzer_type: str = "jieba",
    max_length: int = 4096,
) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)
    _ENSURED.discard((id(client), collection, dim, analyzer_type))
    ensure_collection(
        client,
        collection,
        dim=dim,
        analyzer_type=analyzer_type,
        max_length=max_length,
    )


def upsert_vectors(
    client: Any,
    records: list[VectorRecord],
    collection: str,
) -> None:
    if not records:
        return
    client.upsert(
        collection_name=collection,
        data=[
            {
                "id": int(record.id),
                "text": record.text,
                "dense": list(record.vector),
                "category": record.category,
                "content_type": record.content_type,
                "section_path": record.section_path,
                "source_doc": record.source_doc,
            }
            for record in records
        ],
    )


def dense_search(
    client: Any,
    vector: list[float],
    top_k: int,
    collection: str,
    filter_expr: str | None = None,
) -> list[tuple[int, float]]:
    results = client.search(
        collection_name=collection,
        data=[list(vector)],
        limit=top_k,
        anns_field="dense",
        filter=filter_expr or "",
        output_fields=["id"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        (int(hit["id"]), float(hit["distance"])) for hit in results[0]
    ]


def bm25_search(
    client: Any,
    text: str,
    top_k: int,
    collection: str,
    filter_expr: str | None = None,
) -> list[tuple[int, float]]:
    results = client.search(
        collection_name=collection,
        data=[text],
        limit=top_k,
        anns_field="sparse",
        filter=filter_expr or "",
        output_fields=["id"],
    )
    return [
        (int(hit["id"]), float(hit["distance"])) for hit in results[0]
    ]


def hybrid_search(
    client: Any,
    vector: list[float],
    text: str,
    limit: int,
    collection: str,
    filter_expr: str | None = None,
) -> list[tuple[int, float]]:
    dense_req = AnnSearchRequest(
        data=[list(vector)],
        anns_field="dense",
        param={"metric_type": "COSINE"},
        limit=limit,
        expr=filter_expr,
    )
    sparse_req = AnnSearchRequest(
        data=[text],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=limit,
        expr=filter_expr,
    )
    results = client.hybrid_search(
        collection_name=collection,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(),
        limit=limit,
        output_fields=["id"],
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
