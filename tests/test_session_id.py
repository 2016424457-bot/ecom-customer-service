from app.session_id import MAX_CONVERSATION_ID, parse_conversation_id


def test_decimal_ids_are_canonicalized() -> None:
    assert parse_conversation_id("1") == 1
    assert parse_conversation_id("01") == 1
    assert parse_conversation_id("４２") == 42


def test_non_numeric_values_are_new_conversations() -> None:
    assert parse_conversation_id(None) is None
    assert parse_conversation_id("abc") is None
    assert parse_conversation_id("bad id!") is None
    assert parse_conversation_id("²") is None


def test_out_of_range_ids_are_new_conversations() -> None:
    assert parse_conversation_id(str(MAX_CONVERSATION_ID)) == MAX_CONVERSATION_ID
    assert parse_conversation_id(str(MAX_CONVERSATION_ID + 1)) is None
