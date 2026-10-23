from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from app.graph.build import _builder, build_graph


def _deps():
    return SimpleNamespace(
        settings=SimpleNamespace(max_agent_steps=6),
    )


EXPECTED_NODES = {
    "coref",
    "classify_intent",
    "forced_rag",
    "confidence_check",
    "agent_llm",
    "agent_tools",
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
    assert ("agent_tools", "agent_llm") in edges
    assert ("forced_rag", "confidence_check") in edges
    for node in (
        "complaint_reply",
        "chitchat_reply",
        "fallback_reply",
        "budget_reply",
    ):
        assert (node, "log") in edges
    assert ("log", "__end__") in edges
