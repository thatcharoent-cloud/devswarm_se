"""
devswarm.research.docs_index — SQLite FTS5 full-text documentation index.

Persists web research snippets and fetched pages into a local SQLite database,
allowing local coding agents to query research offline via `search_docs`.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path(".swarm/docs_index.db")


class DocsIndex:
    """SQLite FTS5 full-text search cache for web research documents."""

    def __init__(self, db_path: Path | str = DEFAULT_INDEX_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create documents table and FTS5 virtual index."""
        with self._get_connection() as conn:
            # Metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    retrieved_at TEXT
                )
                """
            )
            # FTS5 index table
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                    url,
                    title,
                    snippet,
                    content,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.commit()

    def insert_document(
        self,
        url: str,
        title: str,
        snippet: str = "",
        content: str = "",
        retrieved_at: str | None = None,
    ) -> None:
        """Upsert a document and update its full-text search entry."""
        retrieved = retrieved_at or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            # Delete old FTS entry if exists
            conn.execute("DELETE FROM docs_fts WHERE url = ?", (url,))
            # Upsert into documents
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (url, title, snippet, content, retrieved_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (url, title, snippet, content, retrieved),
            )
            # Insert into FTS
            conn.execute(
                """
                INSERT INTO docs_fts (url, title, snippet, content)
                VALUES (?, ?, ?, ?)
                """,
                (url, title, snippet, content),
            )
            conn.commit()

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Query FTS5 index for matching documents."""
        # Sanitize query for FTS5 (remove special syntax characters that break FTS)
        clean_query = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
        if not clean_query:
            return []

        # Use OR query or prefix search
        terms = clean_query.split()
        fts_query = " OR ".join(f"{t}*" for t in terms)

        with self._get_connection() as conn:
            try:
                cur = conn.execute(
                    """
                    SELECT d.url, d.title, d.snippet, d.retrieved_at,
                           substr(d.content, 1, 300) AS preview,
                           rank
                    FROM docs_fts f
                    JOIN documents d ON f.url = d.url
                    WHERE docs_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
                rows = cur.fetchall()
                return [
                    {
                        "url": row["url"],
                        "title": row["title"],
                        "snippet": row["snippet"] or row["preview"],
                        "retrieved_at": row["retrieved_at"],
                    }
                    for row in rows
                ]
            except sqlite3.OperationalError as exc:
                logger.warning("FTS search query failed for '%s': %s", query, exc)
                return []

    def get_document(self, url: str) -> dict[str, Any] | None:
        """Retrieve stored document by URL."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM documents WHERE url = ?", (url,))
            row = cur.fetchone()
            if row:
                return dict(row)
            return None

    def count(self) -> int:
        """Return total number of cached documents."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT count(*) FROM documents")
            return int(cur.fetchone()[0])

