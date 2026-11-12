"""
devswarm.research — Host-side web research, documents caching, and source verification.
"""

from devswarm.research.docs_index import DocsIndex
from devswarm.research.search_backend import (
    BraveSearchBackend,
    FakeSearchBackend,
    SearchBackend,
    SearchResult,
    SearXNGSearchBackend,
)
from devswarm.research.tools import (
    ResearchBudgetExceededError,
    ResearchManager,
    SensitiveQueryBlockedError,
)
from devswarm.research.verifier import CitationVerification, SourceVerifier
from devswarm.research.web_fetcher import (
    ContentTypeNotAllowedError,
    FetchError,
    PolicyDomainForbiddenError,
    SSRFViolationError,
    WebFetcher,
)

__all__ = [
    "BraveSearchBackend",
    "CitationVerification",
    "ContentTypeNotAllowedError",
    "DocsIndex",
    "FakeSearchBackend",
    "FetchError",
    "PolicyDomainForbiddenError",
    "ResearchBudgetExceededError",
    "ResearchManager",
    "SSRFViolationError",
    "SearchBackend",
    "SearchResult",
    "SearXNGSearchBackend",
    "SensitiveQueryBlockedError",
    "SourceVerifier",
    "WebFetcher",
]

