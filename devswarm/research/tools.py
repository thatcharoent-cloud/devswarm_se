"""
devswarm.research.tools — Host-side research tools (web_search, fetch_page, search_docs).

Enforces per-run caps (MAX_SEARCHES, MAX_FETCHES, MAX_FETCH_BYTES) and outbound
query secret sanitization.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from devswarm.research.docs_index import DocsIndex
from devswarm.research.search_backend import SearchBackend, SearchResult
from devswarm.research.web_fetcher import WebFetcher

logger = logging.getLogger(__name__)


class ResearchBudgetExceededError(Exception):
    """Raised when web research caps are exceeded."""


class SensitiveQueryBlockedError(Exception):
    """Raised when a search query contains secrets or forbidden patterns."""


@dataclass
class ResearchManager:
    """Manages research tools, budget caps, and caching."""

    backend: SearchBackend
    fetcher: WebFetcher
    docs_index: DocsIndex
    max_searches: int = 10
    max_fetches: int = 15
    max_fetch_bytes: int = 1_000_000

    searches_done: int = 0
    fetches_done: int = 0
    fetch_bytes_done: int = 0

    # Secret patterns blocked from search queries
    secret_patterns: list[re.Pattern[str]] = field(
        default_factory=lambda: [
            re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
            re.compile(r"AIza[0-9A-Za-z-_]{35}"),
            re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
            re.compile(r"(?i)(password|secret|apikey|api_key)\s*[:=]"),
        ]
    )

    def validate_query(self, query: str) -> None:
        """Verify search query does not contain sensitive API keys or credentials."""
        for pat in self.secret_patterns:
            if pat.search(query):
                raise SensitiveQueryBlockedError(
                    "Search query blocked: contains sensitive API key or secret pattern."
                )

    async def web_search(self, query: str, k: int = 5) -> str:
        """Execute web search, enforce caps, index results, and return markdown."""
        if self.searches_done >= self.max_searches:
            raise ResearchBudgetExceededError(
                f"MAX_SEARCHES ({self.max_searches}) reached for this run."
            )

        self.validate_query(query)
        self.searches_done += 1

        results = await self.backend.search(query, k=k)
        if not results:
            return f"No search results found for query: '{query}'"

        now_iso = datetime.now(timezone.utc).isoformat()

        # Cache results into docs_index
        for r in results:
            self.docs_index.insert_document(
                url=r.url,
                title=r.title,
                snippet=r.snippet,
                retrieved_at=now_iso,
            )

        lines = [f"### Web Search Results for: `{query}`\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r.title}]({r.url})\n   {r.snippet}\n")

        return "\n".join(lines)

    async def fetch_page(self, url: str) -> str:
        """Fetch web page, sanitize HTML, wrap in untrusted delimiter, and track bytes."""
        if self.fetches_done >= self.max_fetches:
            raise ResearchBudgetExceededError(
                f"MAX_FETCHES ({self.max_fetches}) reached for this run."
            )

        content = await self.fetcher.fetch(url)
        content_size = len(content.encode("utf-8"))

        if self.fetch_bytes_done + content_size > self.max_fetch_bytes:
            raise ResearchBudgetExceededError(
                f"MAX_FETCH_BYTES ({self.max_fetch_bytes}) exceeded. Total would be {self.fetch_bytes_done + content_size} bytes."
            )

        self.fetches_done += 1
        self.fetch_bytes_done += content_size
        return content

    def search_docs(self, query: str, k: int = 5) -> str:
        """Offline full-text search across all cached documents."""
        hits = self.docs_index.search(query, limit=k)
        if not hits:
            return f"No local cached research documents matched '{query}'."

        lines = [f"### Cached Documentation Matches for: `{query}`\n"]
        for i, hit in enumerate(hits, 1):
            lines.append(
                f"{i}. [{hit['title']}]({hit['url']}) (Retrieved: {hit.get('retrieved_at', 'unknown')})\n"
                f"   {hit['snippet']}\n"
            )
        return "\n".join(lines)

