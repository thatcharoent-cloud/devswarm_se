"""
devswarm.agents.context — Bounded agent context manager with 70% compaction.

Key rules (16 GB profile):
  - NUM_CTX: default 16384 tokens.
  - Compaction threshold: 70% of NUM_CTX (11,468 tokens / ~45,800 characters).
  - Tool output capped at 1500 tokens (~6000 characters).
  - Secret redaction: all tool outputs pass through regex redaction.
  - Compaction strategy: preserve system prompt (turn 0), initial task instructions (turn 1),
    and recent turns (last 4 messages), while compacting older intermediate tool outputs and turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from devswarm.platform.events import redact_secrets

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_NUM_CTX = 16384
DEFAULT_COMPACTION_RATIO = 0.70
MAX_TOOL_OUTPUT_CHARS = 1500 * CHARS_PER_TOKEN  # 6000 characters (1500 tokens)


@dataclass
class AgentMessage:
    """Internal representation of a chat message in the agent conversation."""
    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class AgentContext:
    """Manages chat messages, character/token bounds, and 70% context compaction."""

    def __init__(
        self,
        num_ctx: int = DEFAULT_NUM_CTX,
        compaction_ratio: float = DEFAULT_COMPACTION_RATIO,
        max_tool_chars: int = MAX_TOOL_OUTPUT_CHARS,
    ) -> None:
        self.num_ctx = num_ctx
        self.compaction_ratio = compaction_ratio
        self.compaction_threshold_tokens = int(num_ctx * compaction_ratio)
        self.compaction_threshold_chars = self.compaction_threshold_tokens * CHARS_PER_TOKEN
        self.max_tool_chars = max_tool_chars
        self.messages: list[AgentMessage] = []
        self.compaction_count: int = 0

    def add_system(self, content: str) -> None:
        self.messages.append(AgentMessage(role="system", content=content))

    def add_user(self, content: str) -> None:
        self.messages.append(AgentMessage(role="user", content=content))

    def add_assistant(self, content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.messages.append(
            AgentMessage(
                role="assistant",
                content=content or "",
                tool_calls=tool_calls or [],
            )
        )

    def add_tool_result(self, name: str, result: str, tool_call_id: str | None = None) -> None:
        """Add a tool execution result, applying secret redaction and 1500-token capping."""
        # 1. Redact any sensitive tokens/keys
        clean_result = redact_secrets(result)

        # 2. Cap tool output at 1500 tokens (6000 chars)
        if len(clean_result) > self.max_tool_chars:
            excess = len(clean_result) - self.max_tool_chars
            clean_result = (
                clean_result[:self.max_tool_chars]
                + f"\n... [Tool output truncated: exceeded {self.max_tool_chars // CHARS_PER_TOKEN} token limit by {excess} chars]"
            )

        self.messages.append(
            AgentMessage(
                role="tool",
                name=name,
                content=clean_result,
                tool_call_id=tool_call_id,
            )
        )

    def estimate_tokens(self) -> int:
        """Estimate total tokens across all messages using character heuristic."""
        total_chars = sum(len(m.content) for m in self.messages)
        # Account for tool calls structure
        for m in self.messages:
            if m.tool_calls:
                total_chars += len(str(m.tool_calls))
        return total_chars // CHARS_PER_TOKEN

    def is_compaction_needed(self) -> bool:
        """Return True if estimated token count exceeds 70% of NUM_CTX."""
        return self.estimate_tokens() >= self.compaction_threshold_tokens

    def compact(self) -> bool:
        """Compact conversation history if context threshold is exceeded.

        Preserves:
          - messages[0]: System prompt
          - messages[1]: Initial task description
          - messages[-4:]: Most recent turns (immediate working context)
        Compacts:
          - Middle messages: truncates older tool outputs to brief summaries.
        """
        if not self.is_compaction_needed() or len(self.messages) <= 6:
            return False

        logger.info(
            "Context compaction triggered: %d tokens >= threshold %d (70%% of %d)",
            self.estimate_tokens(),
            self.compaction_threshold_tokens,
            self.num_ctx,
        )

        system_msg = self.messages[0]
        initial_user_msg = self.messages[1]
        recent_msgs = self.messages[-4:]
        middle_msgs = self.messages[2:-4]

        compacted_middle: list[AgentMessage] = []
        for msg in middle_msgs:
            if msg.role == "tool":
                # Condense older tool output to a small summary snippet
                preview = msg.content[:200].replace("\n", " ").strip()
                summary_content = f"[Earlier tool output '{msg.name}' compacted to fit context: {preview}...]"
                compacted_middle.append(AgentMessage(role="tool", name=msg.name, content=summary_content, tool_call_id=msg.tool_call_id))
            elif msg.role == "assistant" and len(msg.content) > 500:
                condensed = msg.content[:300] + " ... [Earlier thoughts condensed]"
                compacted_middle.append(AgentMessage(role="assistant", content=condensed, tool_calls=msg.tool_calls))
            else:
                compacted_middle.append(msg)

        self.messages = [system_msg, initial_user_msg] + compacted_middle + recent_msgs
        self.compaction_count += 1
        logger.info("Context compacted. New token estimate: %d tokens", self.estimate_tokens())
        return True

    def get_messages(self) -> list[dict[str, Any]]:
        """Return formatted message list ready for Ollama /api/chat."""
        self.compact()
        return [m.to_dict() for m in self.messages]

