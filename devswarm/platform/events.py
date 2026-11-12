"""
devswarm.platform.events — Append-only JSONL audit event logger for devswarm runs.

Location: .swarm/runs/<run_id>/events.jsonl
Features:
  - Strict secret redaction (keys, bearer tokens, private keys, passwords).
  - Configurable string and payload truncation to avoid context explosion.
  - Prompt hash tracking (SHA-256).
  - Helper to reconstruct a task's complete chronological history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Patterns matching secrets and API credentials
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),                                    # OpenAI / standard API keys
    re.compile(r"(?i)(api[_-]?key|secret|password|bearer|auth|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.\/+=]{8,})['\"]?"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),                                       # GitHub token
]

DEFAULT_TRUNCATE_CHARS = 2500


def redact_secrets(data: Any) -> Any:
    """Recursively redact secrets from strings, dicts, or lists."""
    if isinstance(data, str):
        redacted = data
        for pattern in SECRET_PATTERNS:
            if pattern.pattern.startswith("(?i)"):
                # Capturing group pattern: keep the key name, redact the secret value
                redacted = pattern.sub(r"\1: [REDACTED_SECRET]", redacted)
            else:
                redacted = pattern.sub("[REDACTED_SECRET]", redacted)
        return redacted

    if isinstance(data, dict):
        new_dict: dict[str, Any] = {}
        for k, v in data.items():
            if any(term in k.lower() for term in ("key", "secret", "password", "token", "auth")):
                new_dict[k] = "[REDACTED_SECRET]"
            else:
                new_dict[k] = redact_secrets(v)
        return new_dict

    if isinstance(data, list):
        return [redact_secrets(item) for item in data]

    return data


def truncate_payload(data: Any, max_chars: int = DEFAULT_TRUNCATE_CHARS) -> Any:
    """Recursively truncate long string fields."""
    if isinstance(data, str):
        if len(data) > max_chars:
            return data[:max_chars] + f" ... [TRUNCATED {len(data) - max_chars} chars]"
        return data

    if isinstance(data, dict):
        return {k: truncate_payload(v, max_chars) for k, v in data.items()}

    if isinstance(data, list):
        return [truncate_payload(item, max_chars) for item in data]

    return data


truncate_output = truncate_payload


def compute_prompt_hash(prompt_text: str) -> str:
    """Return SHA-256 hex digest of a prompt text."""
    if not prompt_text:
        return ""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


@dataclass
class SwarmEvent:
    """Single audit event in a run."""
    run_id: str
    event_type: str
    task_id: str = ""
    role: str = ""
    model: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    tokens_per_sec: float = 0.0
    prompt_hash: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "role": self.role,
            "type": self.event_type,
            "model": self.model,
            "token_usage": self.token_usage,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "prompt_hash": self.prompt_hash,
            "args": truncate_payload(redact_secrets(self.args)),
            "result": truncate_payload(redact_secrets(self.result)),
        }


class EventLog:
    """Manages append-only JSONL event writing for a single run."""

    def __init__(self, run_dir: Path | str, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.events_file = self.run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: str,
        *,
        task_id: str = "",
        role: str = "",
        model: str = "",
        token_usage: dict[str, int] | None = None,
        tokens_per_sec: float = 0.0,
        prompt_hash: str = "",
        args: dict[str, Any] | None = None,
        result: Any = None,
    ) -> SwarmEvent:
        """Create, redact, sanitize, and append an event to the run's events.jsonl."""
        event = SwarmEvent(
            run_id=self.run_id,
            event_type=event_type,
            task_id=task_id,
            role=role,
            model=model,
            token_usage=token_usage or {},
            tokens_per_sec=tokens_per_sec,
            prompt_hash=prompt_hash,
            args=args or {},
            result=result,
        )

        line = json.dumps(event.to_dict(), separators=(",", ":")) + "\n"
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(line)

        return event

    def read_all(self) -> list[dict[str, Any]]:
        """Read all logged events in order."""
        if not self.events_file.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(self.events_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        events.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        continue
        return events

    def reconstruct_task_history(self, task_id: str) -> list[dict[str, Any]]:
        """Reconstruct the complete chronological sequence of events for a task."""
        all_events = self.read_all()
        return [e for e in all_events if e.get("task_id") == task_id]

