from app.db.models import Base


def test_table_and_column_names_match_schema():
    assert set(Base.metadata.tables) == {
        "conversations",
        "messages",
        "faq",
        "tickets",
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
