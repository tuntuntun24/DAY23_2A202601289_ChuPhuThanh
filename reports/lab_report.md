# Day 08 Lab Report

## 1. Student information

- Name: Chu Phú Thành
- Student ID: 2A202601289
- Date: 2026-08-25

## 2. Summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total approval visits | 2 |
| Checkpoint history verified | Yes |

## 3. Architecture

The workflow is a typed LangGraph state machine. Every request passes through intake and an
LLM classifier. Conditional edges select simple answering, tool lookup, clarification, risky
action approval, or bounded error recovery. Tool output is evaluated before answering; failed
output loops through retry and ultimately reaches a dead-letter response. Every path passes
through `finalize` before `END`.

## 4. State schema

Scalar fields such as `route`, `attempt`, `evaluation_result`, `approval`, and `final_answer`
use overwrite semantics because only their latest value is actionable. `messages`,
`tool_results`, `errors`, and `events` use additive reducers so the audit trail survives graph
steps and retry loops. State values remain JSON-serializable for checkpointing.

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Approval visits |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | Yes | 0 | 0 |
| S02_tool | tool | tool | Yes | 0 | 0 |
| S03_missing | missing_info | missing_info | Yes | 0 | 0 |
| S04_risky | risky | risky | Yes | 0 | 1 |
| S05_error | error | error | Yes | 2 | 0 |
| S06_delete | risky | risky | Yes | 0 | 1 |
| S07_dead_letter | error | error | Yes | 1 | 0 |

## 6. Failure analysis

1. A transient tool failure is marked `needs_retry`. The retry node increments `attempt`, and
   routing checks `attempt < max_attempts`; exhausted work goes to dead letter instead of
   creating an unbounded loop.
2. Side-effecting requests are classified as risky and cannot reach the tool until the approval
   node has produced an affirmative decision. Rejection routes to clarification and performs no
   action.
3. Empty or malformed provider output fails explicitly rather than silently fabricating an
   answer. Structured LLM output constrains classification to the five valid routes.

## 7. Persistence / recovery

Each scenario receives a stable `thread_id`, and the compiled graph uses the configured
checkpointer. The scenario runner verifies that state history can be read back and records the
result in `resume_success`. Memory persistence is used by default; SQLite support stores
checkpoints in WAL mode for cross-process recovery. The SQLite behavior is also covered by an
automated persistence test.

## 8. Extension work

SQLite checkpointing and optional real human-in-the-loop interruption are implemented. Setting
`LANGGRAPH_INTERRUPT=true` pauses at approval and accepts a resumed human decision; the default
mock reviewer keeps CI and classroom demonstrations deterministic.

## 9. Improvement plan

Production work would replace the mock tool with authenticated services, add provider timeout
and rate-limit handling, validate authorization separately from approval, add tracing and latency
measurement, and test interrupt/resume recovery against a durable checkpoint database.
