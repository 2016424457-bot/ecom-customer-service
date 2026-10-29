from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from app.graph.build import _builder, build_graph


def _deps():
    return SimpleNamespace(
        settings=SimpleNamespace(max_agent_steps=6),
    )


EXPECTED_NODES = {
    "prepare_context",
    "coref_rewrite",
    "pending_check",
    "tool_intent_gate",
    "classify_intent",
    "other_reply",
    "refund_scope",
    "refund_order_lookup",
    "refund_policy_rag",
    "refund_judge",
    "refund_fallback_reply",
    "forced_rag",
    "confidence_check",
    "agent_llm",
    "agent_tools",
    "ticket_confirm",
    "complaint_reply",
    "chitchat_reply",
    "fallback_reply",
    "budget_reply",
    "log",
}


def test_graph_has_all_nodes():
    graph = _builder(_deps()).compile()
    names = set(graph.get_graph().nodes)
    assert EXPECTED_NODES <= names


def test_graph_compiles_with_checkpointer():
    graph = build_graph(
        checkpointer=InMemorySaver(), deps=_deps()
    )
    assert graph is not None


def test_conditional_edges_have_expected_targets():
    builder = _builder(_deps())
    graph = builder.compile().get_graph()
    edges = {
        (edge.source, edge.target)
        for edge in graph.edges
        if not edge.conditional
    }
    assert ("ticket_confirm", "agent_llm") in edges
    assert ("forced_rag", "confidence_check") in edges
    for node in (
        "complaint_reply",
        "chitchat_reply",
        "fallback_reply",
        "budget_reply",
    ):
        assert (node, "log") in edges
    assert ("log", "__end__") in edges


def test_router_edges_are_wired():
    builder = _builder(_deps())
    graph = builder.compile().get_graph()
    pairs = {(edge.source, edge.target) for edge in graph.edges}

    assert ("__start__", "prepare_context") in pairs
    assert ("prepare_context", "coref_rewrite") in pairs
    assert ("coref_rewrite", "pending_check") in pairs
    for pair in (
        ("pending_check", "refund_order_lookup"),
        ("pending_check", "tool_intent_gate"),
        ("tool_intent_gate", "agent_llm"),
        ("tool_intent_gate", "classify_intent"),
        ("classify_intent", "refund_scope"),
        ("classify_intent", "other_reply"),
        ("refund_scope", "forced_rag"),
        ("refund_scope", "refund_order_lookup"),
        ("refund_order_lookup", "refund_policy_rag"),
        ("refund_order_lookup", "log"),
        ("refund_policy_rag", "refund_judge"),
        ("refund_policy_rag", "refund_fallback_reply"),
        ("refund_judge", "log"),
        ("refund_fallback_reply", "log"),
        ("other_reply", "log"),
        ("agent_tools", "ticket_confirm"),
        ("agent_tools", "agent_llm"),
        ("ticket_confirm", "agent_llm"),
    ):
        assert pair in pairs
