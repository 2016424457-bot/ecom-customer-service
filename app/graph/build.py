from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.deps import GraphDeps
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


def _builder(deps: GraphDeps) -> StateGraph:
    builder = StateGraph(ConversationState)

    builder.add_node("coref", partial(nodes.coref, deps=deps))
    builder.add_node(
        "classify_intent", partial(nodes.classify_intent, deps=deps)
    )
    builder.add_node("forced_rag", partial(nodes.forced_rag, deps=deps))
    builder.add_node(
        "confidence_check", partial(nodes.confidence_check, deps=deps)
    )
    builder.add_node("agent_llm", partial(nodes.agent_llm, deps=deps))
    builder.add_node("agent_tools", partial(nodes.agent_tools, deps=deps))
    builder.add_node(
        "complaint_reply", partial(nodes.complaint_reply, deps=deps)
    )
    builder.add_node(
        "chitchat_reply", partial(nodes.chitchat_reply, deps=deps)
    )
    builder.add_node(
        "fallback_reply", partial(nodes.fallback_reply, deps=deps)
    )
    builder.add_node("budget_reply", partial(nodes.budget_reply, deps=deps))
    builder.add_node("log", partial(nodes.log_node, deps=deps))

    builder.add_edge(START, "coref")
    builder.add_edge("coref", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "knowledge": "forced_rag",
            "business": "agent_llm",
            "complaint": "complaint_reply",
            "chitchat": "chitchat_reply",
        },
    )

    builder.add_edge("forced_rag", "confidence_check")
    builder.add_conditional_edges(
        "confidence_check",
        confidence_gate,
        {"strong": "agent_llm", "weak": "fallback_reply"},
    )

    max_steps = deps.settings.max_agent_steps

    def _should_continue(state: dict) -> str:
        return should_continue(state, max_steps=max_steps)

    builder.add_conditional_edges(
        "agent_llm",
        _should_continue,
        {
            "continue": "agent_tools",
            "stop": "log",
            "cap": "budget_reply",
        },
    )
    builder.add_edge("agent_tools", "agent_llm")

    for reply in (
        "complaint_reply",
        "chitchat_reply",
        "fallback_reply",
        "budget_reply",
    ):
        builder.add_edge(reply, "log")
    builder.add_edge("log", END)

    return builder


def build_graph(checkpointer: Any, deps: GraphDeps):
    """Compile the CH05 graph with a checkpointer and per-turn deps."""
    return _builder(deps).compile(checkpointer=checkpointer)
