"""
devswarm.tools.registry — Tool registration and schema generator for Ollama function calling.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from devswarm.providers.base import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass
class SwarmTool:
    """Registered tool definition."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert to Ollama tool specification format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_tool_definition(self) -> ToolDefinition:
        """Convert to normalized ToolDefinition for Provider layer."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolRegistry:
    """Central registry of agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, SwarmTool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register a new tool."""
        self._tools[name] = SwarmTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def register_research_tools(self, research_manager: Any) -> None:
        """Register web research and local document cache search tools."""
        self.register(
            name="web_search",
            description="Search the web for technical specifications, documentation, and library references.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "k": {"type": "integer", "description": "Number of results to return (default 5)."},
                },
                "required": ["query"],
            },
            handler=research_manager.web_search,
        )
        self.register(
            name="fetch_page",
            description="Fetch a web page by URL, extract clean readable text, and cache it offline.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The HTTP or HTTPS URL to fetch."},
                },
                "required": ["url"],
            },
            handler=research_manager.fetch_page,
        )
        self.register(
            name="search_docs",
            description="Full-text search over previously retrieved web research and documentation cached offline.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query terms."},
                    "k": {"type": "integer", "description": "Maximum number of document snippets to return."},
                },
                "required": ["query"],
            },
            handler=research_manager.search_docs,
        )

    def get_tool(self, name: str) -> SwarmTool | None:
        return self._tools.get(name)

    def get_tools_for_role(self, allowed_tool_names: list[str] | set[str] | None = None) -> list[dict[str, Any]]:
        """Return Ollama tool definitions filtered by allowed tool names."""
        if allowed_tool_names is None:
            return [t.to_ollama_tool() for t in self._tools.values()]

        allowed = set(allowed_tool_names)
        return [
            t.to_ollama_tool()
            for name, t in self._tools.items()
            if name in allowed
        ]

    def get_tool_definitions_for_role(
        self, allowed_tool_names: list[str] | set[str] | None = None
    ) -> list[ToolDefinition]:
        """Return normalized ToolDefinition instances filtered by allowed tool names."""
        if allowed_tool_names is None:
            return [t.to_tool_definition() for t in self._tools.values()]

        allowed = set(allowed_tool_names)
        return [
            t.to_tool_definition()
            for name, t in self._tools.items()
            if name in allowed
        ]

    async def execute(self, name: str, kwargs: dict[str, Any]) -> str:
        """Execute a tool by name with arguments."""
        tool = self.get_tool(name)
        if not tool:
            return f"<UNTRUSTED_TOOL_DATA status=\"ERROR\">\nError: Tool '{name}' is not recognized.\n</UNTRUSTED_TOOL_DATA>"

        try:
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**kwargs)
            else:
                result = tool.handler(**kwargs)

            return str(result)
        except Exception as exc:
            logger.exception("Error executing tool %s: %s", name, exc)
            return f"<UNTRUSTED_TOOL_DATA status=\"ERROR\">\nTool execution error: {exc}\n</UNTRUSTED_TOOL_DATA>"

