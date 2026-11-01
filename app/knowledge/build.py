from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from app.config import Settings
from app.knowledge.chunking import build_chunks
from app.knowledge.embeddings import embed_texts
from app.knowledge.milvus_client import (
    acall,
    delete_vectors,
    ensure_collection,
    get_client,
    has_ids,
    upsert_vectors,
)
from app.knowledge.sources import FAQ_SOURCE
from app.knowledge.types import Chunk, VectorRecord

Embedder = Callable[
    [list[str], Settings], Awaitable[list[list[float]]]
]


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None:
        ...

    def delete(self, ids: list[int]) -> int:
        ...

    def has_ids(self, ids: list[int]) -> set[int]:
        ...


class MilvusStore:
    def __init__(self, settings: Settings) -> None:
        self.collection = settings.milvus_collection
        self.client = get_client(settings)
        ensure_collection(
            self.client,
            self.collection,
            dim=settings.embed_dim,
            analyzer_type=settings.milvus_analyzer_type,
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        upsert_vectors(self.client, records, self.collection)

    def delete(self, ids: list[int]) -> int:
        return delete_vectors(self.client, ids, self.collection)

    def has_ids(self, ids: list[int]) -> set[int]:
        return has_ids(self.client, ids, self.collection)


def _resolve_store(
    store: VectorStore | None, settings: Settings
) -> VectorStore:
    return store if store is not None else MilvusStore(settings)


async def write_pending(
    repository: Any, chunks: list[Chunk]
) -> list[int]:
    ids: list[int] = []
    for chunk in chunks:
        chunk_id = await repository.insert_knowledge_chunk(
            category=chunk.category,
            questions=chunk.questions,
            answer=chunk.answer,
            section_path=chunk.section_path,
            content_type=chunk.content_type,
            is_key_clause=chunk.is_key_clause,
            source_doc=chunk.source_doc,
        )
        ids.append(chunk_id)

    for index, chunk_id in enumerate(ids):
        previous_id = ids[index - 1] if index > 0 else None
        next_id = ids[index + 1] if index + 1 < len(ids) else None
        await repository.set_chunk_neighbors(
            chunk_id, previous_id, next_id
        )
    return ids


async def build_document(
    repository: Any,
    path: Path,
    *,
    settings: Settings,
    content_type: str,
    reset: bool = False,
    store: VectorStore | None = None,
) -> int:
    source_doc = path.name
    markdown = path.read_text(encoding="utf-8")
    chunks = build_chunks(
        markdown,
        content_type=content_type,
        source_doc=source_doc,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        table_max_rows=settings.table_max_rows,
    )

    if reset:
        await reset_source(repository, source_doc, settings, store=store)
    elif await repository.count_chunks_by_source(source_doc) > 0:
        return 0

    ids = await write_pending(repository, chunks)
    await repository.commit()
    return len(ids)


async def build_faq(
    repository: Any,
    settings: Settings,
    *,
    reset: bool = False,
    store: VectorStore | None = None,
) -> int:
    rows = await repository.list_faq_rows()
    chunks = [
        Chunk(
            category=row.category,
            questions=row.question,
            answer=row.answer,
            section_path=None,
            content_type="faq",
            is_key_clause=False,
            source_doc=FAQ_SOURCE,
        )
        for row in rows
    ]

    if reset:
        await reset_source(repository, FAQ_SOURCE, settings, store=store)
    elif await repository.count_chunks_by_source(FAQ_SOURCE) > 0:
        return 0

    ids = await write_pending(repository, chunks)
    await repository.commit()
    return len(ids)


async def vectorize_pending(
    repository: Any,
    settings: Settings,
    *,
    batch_size: int = 16,
    limit: int | None = None,
    store: VectorStore | None = None,
    embedder: Embedder = embed_texts,
) -> int:
    pending = await repository.list_pending_chunks(limit=limit)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return 0

    vector_store = _resolve_store(store, settings)
    done = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [
            f"{row.category}\n{row.questions}\n{row.answer}"
            for row in batch
        ]
        vectors = await embedder(texts, settings)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding count {len(vectors)} != chunk count {len(batch)}"
            )
        for vector in vectors:
            if len(vector) != settings.embed_dim:
                raise RuntimeError(
                    f"embedding dim {len(vector)} != {settings.embed_dim}"
                )

        records = [
            VectorRecord(
                id=row.id,
                text=text,
                vector=vector,
                category=row.category,
                content_type=row.content_type,
                section_path=row.section_path,
                source_doc=row.source_doc,
            )
            for row, text, vector in zip(
                batch, texts, vectors, strict=True
            )
        ]
        await acall(vector_store.upsert, records)
        for row in batch:
            await repository.mark_chunk_vectorized(row.id, str(row.id))
        await repository.commit()
        done += len(batch)
    return done


async def vectorize_chunk(
    repository: Any,
    settings: Settings,
    chunk_id: int,
    *,
    store: VectorStore | None = None,
    embedder: Embedder = embed_texts,
) -> None:
    row = await repository.get_chunk_by_id(chunk_id)
    if row is None:
        raise ValueError(f"knowledge chunk {chunk_id} not found")

    text = f"{row.category}\n{row.questions}\n{row.answer}"
    vectors = await embedder([text], settings)
    if len(vectors) != 1:
        raise RuntimeError(
            f"embedding count {len(vectors)} != chunk count 1"
        )
    vector = vectors[0]
    if len(vector) != settings.embed_dim:
        raise RuntimeError(
            f"embedding dim {len(vector)} != {settings.embed_dim}"
        )

    vector_store = _resolve_store(store, settings)
    record = VectorRecord(
        id=row.id,
        text=text,
        vector=vector,
        category=row.category,
        content_type=row.content_type,
        section_path=row.section_path,
        source_doc=row.source_doc,
    )
    await acall(vector_store.upsert, [record])
    await repository.mark_chunk_vectorized(row.id, str(row.id))
    await repository.commit()


async def verify_vectors(
    repository: Any,
    settings: Settings,
    *,
    store: VectorStore | None = None,
) -> int:
    done_rows = await repository.list_done_chunks()
    if not done_rows:
        return 0

    vector_store = _resolve_store(store, settings)
    ids = [row.id for row in done_rows]
    existing = await acall(vector_store.has_ids, ids)
    missing = [chunk_id for chunk_id in ids if chunk_id not in existing]
    if missing:
        await repository.mark_chunks_pending(missing)
        await repository.commit()
    return len(missing)


async def requeue_done_chunks(repository: Any) -> int:
    rows = await repository.list_done_chunks()
    ids = [row.id for row in rows]
    if not ids:
        return 0
    await repository.mark_chunks_pending(ids)
    await repository.commit()
    return len(ids)


async def reset_source(
    repository: Any,
    source_doc: str,
    settings: Settings,
    *,
    store: VectorStore | None = None,
) -> int:
    ids = await repository.delete_chunks_by_source(source_doc)
    if not ids:
        return 0
    # Commit the MySQL delete first: if the process dies before the Milvus
    # delete, the only damage is an orphan vector (ignored by retrieval),
    # never a live `done` row whose vector is gone.
    await repository.commit()
    vector_store = _resolve_store(store, settings)
    await acall(vector_store.delete, ids)
    return len(ids)
