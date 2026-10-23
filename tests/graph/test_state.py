from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.graph.deps import GraphDeps
from app.graph.state import ConversationState, merge_dict, merge_trace


def test_merge_dict_accumulates():
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dict(None, {"b": 2}) == {"b": 2}
    assert merge_dict({"a": 1}, {"a": 9}) == {"a": 9}


def test_merge_trace_merges_into_state_trace():
    assert merge_trace({"trace": {"a": 1}}, b=2) == {"a": 1, "b": 2}
    assert merge_trace({}, forced_rag=True) == {"forced_rag": True}


def test_add_messages_reducer_appends():
    merged = add_messages([HumanMessage("hi")], [AIMessage("yo")])
    assert [message.content for message in merged] == ["hi", "yo"]


def test_state_has_required_keys():
    keys = ConversationState.__annotations__
    for key in [
        "messages",
        "user_id",
        "conversation_id",
        "intent",
        "route",
        "evidence",
        "citations",
        "evidence_strong",
        "answer",
        "steps",
        "tokens_used",
        "suggested_actions",
        "trace",
    ]:
        assert key in keys, f"missing state key {key}"


def test_graph_deps_exposes_node_dependencies():
    fields = GraphDeps.__dataclass_fields__
    for name in [
        "settings",
        "chat_model",
        "intent_classifier",
        "prompt_builder",
        "repository",
        "knowledge_repository",
        "evidence_checker",
        "faq_search",
        "tools",
        "tool_registry",
        "agent_prompt",
    ]:
        assert name in fields, f"missing GraphDeps field {name}"
