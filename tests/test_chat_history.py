from langchain_core.messages import AIMessage, ToolMessage

from app.services.chat_history import StoredMessage, to_langchain


def test_assistant_tool_calls_round_trip():
    rows = [
        StoredMessage("user", "查物流", None, None),
        StoredMessage(
            "assistant",
            None,
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "query_logistics",
                        "arguments": '{"order_id":"1001"}',
                    },
                }
            ],
            None,
        ),
        StoredMessage("tool", '{"status":"运输中"}', None, "call_1"),
    ]

    messages = to_langchain(rows)

    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "query_logistics"
    assert messages[1].tool_calls[0]["args"] == {"order_id": "1001"}
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_1"


def test_missing_tool_response_is_repaired():
    rows = [
        StoredMessage("user", "查物流", None, None),
        StoredMessage(
            "assistant",
            None,
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "query_logistics",
                        "arguments": "{}",
                    },
                }
            ],
            None,
        ),
    ]

    messages = to_langchain(rows)

    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].tool_call_id == "call_1"
    assert "incomplete_turn" in messages[-1].content


def test_orphan_tool_message_is_dropped():
    rows = [StoredMessage("tool", "{}", None, "call_missing")]

    assert to_langchain(rows) == []


def test_invalid_arguments_json_becomes_empty_dict():
    rows = [
        StoredMessage(
            "assistant",
            None,
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "query_logistics",
                        "arguments": "{not json",
                    },
                }
            ],
            None,
        )
    ]

    messages = to_langchain(rows)

    assert messages[0].tool_calls[0]["args"] == {}
