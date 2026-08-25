"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event

_PRIMARY_RATE_LIMITED = False


class Classification(BaseModel):
    """Structured output returned by the classification LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "high"] = "low"
    rationale: str = Field(description="Brief reason for selecting the route")


def _text_content(response: object) -> str:
    """Normalize LangChain provider response content to plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content).strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _fallback_model() -> str:
    return os.getenv("LLM_FALLBACK_MODEL", "gemini-3.5-flash-lite")


def _primary_rate_limited() -> bool:
    return _PRIMARY_RATE_LIMITED


def _mark_primary_rate_limited() -> None:
    global _PRIMARY_RATE_LIMITED
    _PRIMARY_RATE_LIMITED = True


def _available_llm() -> BaseChatModel:
    if _primary_rate_limited():
        return get_llm(model=_fallback_model(), temperature=0)
    return get_llm(temperature=0)


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# Workflow nodes


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    prompt = f"""You route customer-support tickets. Classify the request into exactly one route.

Routes:
- risky: explicitly asks the agent/system to PERFORM a side effect now, including refund,
  delete, cancel, modify stored data, send an email/message, or charge money. A how-to question
  that merely mentions such an operation is not risky. Risky takes priority over other routes.
- tool: asks only to retrieve or look up specific information such as order/tracking status.
- missing_info: too vague or incomplete to identify the issue or action.
- error: reports a timeout, crash, unavailable service, or other system/process failure.
- simple: a general how-to or support question answerable without a tool or side effect.

Priority when multiple apply: risky > tool > missing_info > error > simple.
Set risk_level=high only for risky; otherwise low.
Classify by what the customer asks the agent to execute, not by isolated verbs or nouns.
Example: "How do I reset my password?" is simple; "Reset my password now" is risky.

Customer request: {state.get("query", "")!r}
"""
    using_fallback = _primary_rate_limited()
    classifier = _available_llm().with_structured_output(Classification)
    try:
        result = classifier.invoke(prompt)
    except Exception as exc:
        if using_fallback or not os.getenv("GEMINI_API_KEY") or not _is_rate_limit_error(exc):
            raise
        _mark_primary_rate_limited()
        fallback = get_llm(model=_fallback_model(), temperature=0)
        result = fallback.with_structured_output(Classification).invoke(prompt)
    parsed = Classification.model_validate(result)
    risk_level = "high" if parsed.route == "risky" else "low"
    return {
        "route": parsed.route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {parsed.route}",
                rationale=parsed.rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient support service failure on attempt {attempt}"
        event_type = "failed"
    elif route == "risky":
        result = f"Approved action completed successfully: {state.get('proposed_action') or query}"
        event_type = "completed"
    else:
        result = f"Support lookup completed successfully for request: {query}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    results = state.get("tool_results", [])
    latest = results[-1] if results else "ERROR: tool returned no result"
    evaluation_result = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"tool result evaluation: {evaluation_result}",
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    prompt = f"""You are a concise, helpful customer-support assistant.
Answer the customer's request using only the supplied context. Do not invent account,
order, policy, or tool details. If no tool context exists, give safe general guidance.
If an approved action was completed, clearly state that outcome.

Customer request: {state.get("query", "")}
Route: {state.get("route", "")}
Tool context: {tool_results if tool_results else "No tool was needed."}
Approval context: {approval if approval is not None else "Not applicable."}
"""
    using_fallback = _primary_rate_limited()
    try:
        response = _available_llm().invoke(prompt)
    except Exception as exc:
        if using_fallback or not os.getenv("GEMINI_API_KEY") or not _is_rate_limit_error(exc):
            raise
        _mark_primary_rate_limited()
        response = get_llm(model=_fallback_model(), temperature=0).invoke(prompt)
    final_answer = _text_content(response)
    if not final_answer:
        raise RuntimeError("LLM returned an empty support answer")
    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", "grounded answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    if state.get("route") == "risky" and state.get("approval") is not None:
        question = (
            "The requested action was not approved. Would you like a non-destructive "
            "alternative, or can you provide revised authorization?"
        )
    else:
        question = (
            f"Could you provide the affected account, order, or feature and describe what "
            f"you expected to happen? Your current request was: {query!r}."
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    proposed_action = state.get("query", "").strip()
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "approval_required",
                "side-effecting action prepared for human approval",
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return an approval decision and its audit event.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        resumed = interrupt(
            {
                "question": "Approve this side-effecting support action?",
                "proposed_action": state.get("proposed_action", ""),
            }
        )
        if isinstance(resumed, dict):
            decision = ApprovalDecision.model_validate(resumed)
        else:
            decision = ApprovalDecision(
                approved=bool(resumed),
                reviewer="human-reviewer",
                comment="Decision received through LangGraph interrupt",
            )
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Automatically approved for deterministic lab execution",
        )
    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                "approved" if decision.approved else "rejected",
                decision.comment,
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    error = f"Transient failure recorded; retry attempt {attempt}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "scheduled", error, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempts = int(state.get("attempt", 0))
    final_answer = (
        f"We could not complete the request after {attempts} attempt(s). "
        "It has been escalated for manual support review."
    )
    return {
        "final_answer": final_answer,
        "events": [make_event("dead_letter", "escalated", final_answer)],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
