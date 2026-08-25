"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.types import Checkpointer


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Checkpointer:
    """Return a LangGraph checkpointer.

    Memory, SQLite, and Postgres backends are supported.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        database_path = database_url or "checkpoints.db"
        if database_path.startswith("sqlite:///"):
            database_path = database_path.removeprefix("sqlite:///")
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn=connection)
        saver.setup()
        return saver
    if kind == "postgres":
        if not database_url:
            raise ValueError("database_url is required for the Postgres checkpointer")
        try:
            import psycopg  # type: ignore[import-not-found]
            from langgraph.checkpoint.postgres import (  # type: ignore[import-not-found]
                PostgresSaver,
            )
        except ImportError as exc:
            raise RuntimeError("Install the project with the 'postgres' extra") from exc
        connection = psycopg.connect(database_url, autocommit=True, prepare_threshold=0)
        saver = PostgresSaver(conn=connection)
        saver.setup()
        return saver
    raise ValueError(f"Unknown checkpointer kind: {kind}")
