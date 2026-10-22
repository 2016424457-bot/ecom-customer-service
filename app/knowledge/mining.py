import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.knowledge.build import vectorize_pending
from app.knowledge.dedup import dedupe_indices
from app.knowledge.schemas import parse_extraction
from app.knowledge.types import QaPair, StagingRow


class MiningError(RuntimeError):
    def __init__(self, conversation_id: int) -> None:
        super().__init__(
            f"mining failed for conversation {conversation_id}"
        )
        self.conversation_id = conversation_id


@dataclass
class MiningStats:
    conversations: int = 0
    extracted: int = 0
    kept: int = 0
    discarded: int = 0
    vectorized: int = 0


def format_transcript(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        role = "用户" if message.role == "user" else "客服"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


async def extract_qas(
    transcript: str, mining_model: Any, prompt: str
) -> list[QaPair]:
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=transcript),
    ]
    result = await mining_model.ainvoke(messages)
    return parse_extraction(result)


async def finalize_staging(
    repository: Any,
    settings: Settings,
    *,
    batch_no: str | None = None,
    sync: bool = True,
) -> MiningStats:
    rows = await repository.list_staging_by_status(
        "extracted", batch_no=batch_no
    )
    if not rows:
        stats = MiningStats()
        if sync:
            # A previous run may have left pending chunks (for example after
            # an embedding failure); retry them even when nothing new was
            # extracted.
            stats.vectorized = await vectorize_pending(
                repository, settings
            )
        return stats

    pairs = [
        QaPair(
            row.question,
            row.answer,
            row.category or "历史对话",
        )
        for row in rows
    ]
    existing = await repository.list_all_questions()
    kept_indices, dropped_indices = dedupe_indices(pairs, existing)
    kept_rows = [rows[index] for index in kept_indices]
    dropped_rows = [rows[index] for index in dropped_indices]

    # Insert chunks before marking kept: a crash in between leaves the rows
    # extracted, and the next run discards them because the question now
    # exists in the knowledge base.
    for row in kept_rows:
        await repository.insert_knowledge_chunk(
            category=row.category or "历史对话",
            questions=row.question,
            answer=row.answer,
            section_path=None,
            content_type="qa",
            is_key_clause=False,
            source_doc=row.source_ref or "conversation:unknown",
        )
    await repository.commit()

    await repository.set_staging_status(
        [row.id for row in kept_rows], "kept"
    )
    await repository.set_staging_status(
        [row.id for row in dropped_rows], "discarded"
    )
    await repository.commit()

    stats = MiningStats(
        kept=len(kept_rows), discarded=len(dropped_rows)
    )
    if sync:
        stats.vectorized = await vectorize_pending(repository, settings)
    return stats


async def mine_once(
    repository: Any,
    settings: Settings,
    mining_model: Any,
    prompt: str,
    *,
    batch_no: str | None = None,
    sync: bool = True,
) -> MiningStats:
    batch = batch_no or (
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    stats = MiningStats()
    cursor = await repository.get_mining_cursor()
    windows = await repository.list_conversations_to_mine(
        cursor, settings.mining_batch_conversations
    )

    for window in windows:
        messages = await repository.list_conversation_messages(
            window.conversation_id,
            window.max_message_id,
            settings.mining_max_messages,
        )
        transcript = format_transcript(messages)
        if not transcript:
            await repository.set_mining_cursor(window.max_message_id)
            await repository.commit()
            continue
        try:
            pairs = await extract_qas(
                transcript, mining_model, prompt
            )
        except Exception as exc:
            raise MiningError(window.conversation_id) from exc

        rows = [
            StagingRow(
                batch_no=batch,
                source_ref=f"conversation:{window.conversation_id}",
                category=pair.category,
                question=pair.question,
                answer=pair.answer,
            )
            for pair in pairs
        ]
        await repository.insert_staging_rows(rows)
        await repository.set_mining_cursor(window.max_message_id)
        await repository.commit()
        stats.conversations += 1
        stats.extracted += len(rows)

    finalized = await finalize_staging(
        repository, settings, sync=sync
    )
    stats.kept += finalized.kept
    stats.discarded += finalized.discarded
    stats.vectorized += finalized.vectorized
    return stats
