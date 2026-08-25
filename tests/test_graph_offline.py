"""Deterministic graph coverage without consuming an external LLM quota."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state, make_event


class _StructuredFake:
    def invoke(self, prompt: str) -> dict[str, str]:
        text = prompt.lower()
        if "refund this customer" in text or "delete customer account" in text:
            route = "risky"
        elif "lookup order status" in text:
            route = "tool"
        elif "can you fix it" in text:
            route = "missing_info"
        elif "timeout failure" in text or "system failure" in text:
            route = "error"
        else:
            route = "simple"
        return {
            "route": route,
            "risk_level": "high" if route == "risky" else "low",
            "rationale": "deterministic test classification",
        }


class _FakeLlm:
    def with_structured_output(self, _schema: type[Any]) -> _StructuredFake:
        return _StructuredFake()

    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content="Grounded test response")


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "_available_llm", lambda: _FakeLlm())


def test_all_sample_scenarios_terminate_with_expected_routes(fake_llm: None) -> None:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenarios = [
        Scenario(id="simple", query="How do I reset my password?", expected_route=Route.SIMPLE),
        Scenario(id="tool", query="Please lookup order status 123", expected_route=Route.TOOL),
        Scenario(id="missing", query="Can you fix it?", expected_route=Route.MISSING_INFO),
        Scenario(id="risky", query="Refund this customer", expected_route=Route.RISKY),
        Scenario(id="error", query="Timeout failure while processing", expected_route=Route.ERROR),
    ]

    for scenario in scenarios:
        state = initial_state(scenario)
        config = {"configurable": {"thread_id": state["thread_id"]}}
        result = graph.invoke(state, config=config)
        assert result["route"] == scenario.expected_route.value
        assert result.get("final_answer") or result.get("pending_question")
        assert result["events"][-1]["node"] == "finalize"


def test_rejected_risky_action_never_calls_tool(
    fake_llm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "approval": {"approved": False, "reviewer": "test", "comment": "rejected"},
            "events": [make_event("approval", "rejected", "rejected")],
        }

    monkeypatch.setattr(nodes, "approval_node", reject)
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="reject", query="Refund this customer", expected_route=Route.RISKY)
    state = initial_state(scenario)
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )

    assert result["approval"]["approved"] is False
    assert result["tool_results"] == []
    assert result["pending_question"]


def test_sqlite_checkpoint_history_survives_rebuild(
    fake_llm: None, tmp_path: Any
) -> None:
    database_path = tmp_path / "checkpoints.db"
    checkpointer = build_checkpointer("sqlite", str(database_path))
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(id="persist", query="How do I get help?", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}

    graph.invoke(state, config=config)
    assert list(graph.get_state_history(config))
    checkpointer.conn.close()

    reopened = build_checkpointer("sqlite", str(database_path))
    rebuilt_graph = build_graph(checkpointer=reopened)
    try:
        latest = rebuilt_graph.get_state(config)
        assert latest.values["scenario_id"] == "persist"
        assert latest.values["final_answer"] == "Grounded test response"
    finally:
        reopened.conn.close()
