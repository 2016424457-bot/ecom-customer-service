MAX_CONVERSATION_ID = 2**64 - 1


def parse_conversation_id(session_id: str | None) -> int | None:
    """Return the canonical BIGINT UNSIGNED conversation id, if any.

    Only decimal digit strings inside MySQL's BIGINT UNSIGNED range are
    accepted. Everything else is a new-conversation request.
    """
    if not session_id or not session_id.isdecimal():
        return None
    try:
        value = int(session_id)
    except ValueError:
        return None
    if 0 <= value <= MAX_CONVERSATION_ID:
        return value
    return None
