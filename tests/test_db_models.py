from app.db.models import Base


def test_table_and_column_names_match_schema():
    assert set(Base.metadata.tables) == {
        "conversations",
        "messages",
        "faq",
        "tickets",
        "knowledge_chunks",
        "qa_extraction_staging",
        "knowledge_mining_state",
        "low_confidence_questions",
        "faith_cases",
    }
    assert [c.name for c in Base.metadata.tables["conversations"].columns] == [
        "id",
        "user_id",
        "status",
        "created_at",
        "updated_at",
    ]
    assert [c.name for c in Base.metadata.tables["messages"].columns] == [
        "id",
        "conversation_id",
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "created_at",
    ]
    assert [c.name for c in Base.metadata.tables["faq"].columns] == [
        "id",
        "question",
        "answer",
        "category",
        "created_at",
        "updated_at",
    ]
    assert [c.name for c in Base.metadata.tables["tickets"].columns] == [
        "ticket_no",
        "conversation_id",
        "description",
        "ticket_type",
        "status",
        "created_at",
    ]
    assert [c.name for c in Base.metadata.tables["knowledge_chunks"].columns] == [
        "id",
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
        "prev_chunk_id",
        "next_chunk_id",
        "vector_id",
        "vectorize_status",
        "source_doc",
        "created_at",
        "updated_at",
    ]
    assert [
        c.name
        for c in Base.metadata.tables["qa_extraction_staging"].columns
    ] == [
        "id",
        "batch_no",
        "source_ref",
        "category",
        "question",
        "answer",
        "status",
        "created_at",
    ]
    assert [
        c.name for c in Base.metadata.tables["knowledge_mining_state"].columns
    ] == ["id", "last_message_id", "updated_at"]


def test_ch04_tables_and_columns():
    assert "low_confidence_questions" in Base.metadata.tables
    assert [
        c.name
        for c in Base.metadata.tables["low_confidence_questions"].columns
    ] == [
        "id",
        "conversation_id",
        "raw_question",
        "source",
        "reason",
        "created_at",
    ]
    assert "faith_cases" in Base.metadata.tables
    assert [
        c.name for c in Base.metadata.tables["faith_cases"].columns
    ] == [
        "id",
        "eval_id",
        "bucket",
        "query",
        "strategy",
        "answer",
        "reason",
        "citations",
        "judge_model",
        "status",
        "seen_count",
        "first_seen_at",
        "last_seen_at",
        "resolution",
        "resolved_at",
    ]
