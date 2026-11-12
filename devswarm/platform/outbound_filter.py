"""
devswarm.platform.outbound_filter — Single security gatekeeper for all outbound cloud calls.

Enforces:
  1. Explicit first-call cloud acknowledgment (.swarm/cloud_ack.json).
  2. Role-based content class allowlists.
  3. Hard deny list against sensitive files (.env, keys, certs, .git, .swarm).
  4. Automatic redaction of API keys, tokens, and private secrets.
  5. Maximum payload byte size caps.
  6. Append-only outbound audit logging (.swarm/runs/<run_id>/outbound_audit.jsonl).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from devswarm.providers.base import ChatMessage

logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = Path("config/policy.yaml")
DEFAULT_SWARM_DIR = Path(".swarm")


class SecurityPolicyViolationError(Exception):
    """Raised when an outbound payload violates security policy."""


class CloudConfirmationRequiredError(SecurityPolicyViolationError):
    """Raised when the user has not confirmed cloud usage for the repository."""


class PayloadSizeExceededError(SecurityPolicyViolationError):
    """Raised when payload exceeds max_payload_bytes."""


class OfflineModeActiveError(Exception):
    """Raised when cloud calls are attempted while --offline is active."""


class OutboundFilter:
    """Security filter through which every cloud LLM request must pass."""

    def __init__(
        self,
        policy_path: Path | str = DEFAULT_POLICY_PATH,
        swarm_dir: Path | str = DEFAULT_SWARM_DIR,
        offline: bool = False,
        cloud_sees_code: bool = False,
        log_payloads: bool = False,
        confirmation_callback: Callable[[str, str, str, list[str]], bool] | None = None,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.swarm_dir = Path(swarm_dir)
        self.offline = offline
        self.cloud_sees_code = cloud_sees_code
        self.log_payloads = log_payloads
        self.confirmation_callback = confirmation_callback

        self.max_payload_bytes = 65536
        self.hard_deny_patterns: list[str] = []
        self.allowed_classes_by_role: dict[str, list[str]] = {}
        self.redaction_regexes: list[re.Pattern[str]] = []

        self._load_policy()

    def _load_policy(self) -> None:
        if not self.policy_path.exists():
            return

        try:
            data = yaml.safe_load(self.policy_path.read_text(encoding="utf-8")) or {}
            cp = data.get("cloud_data_policy", {})
            self.max_payload_bytes = int(cp.get("max_payload_bytes", 65536))
            self.hard_deny_patterns = list(cp.get("hard_deny_patterns", []))
            self.allowed_classes_by_role = cp.get("allowed_content_classes_per_role", {})

            raw_redactions = cp.get("redaction_patterns", [])
            self.redaction_regexes = []
            for pat in raw_redactions:
                try:
                    self.redaction_regexes.append(re.compile(pat))
                except re.error as err:
                    logger.warning("Invalid regex in policy redaction patterns: %s (%s)", pat, err)
        except Exception as exc:
            logger.warning("Failed loading cloud policy from %s: %s", self.policy_path, exc)

    def check_cloud_acknowledgment(self, provider: str, model: str, role: str, content_classes: list[str]) -> bool:
        """Verify that human confirmed cloud calls for this repo, or request it."""
        ack_file = self.swarm_dir / "cloud_ack.json"
        if ack_file.exists():
            return True

        if self.confirmation_callback:
            approved = self.confirmation_callback(provider, model, role, content_classes)
            if approved:
                self.record_acknowledgment(provider, model)
                return True

        raise CloudConfirmationRequiredError(
            f"Explicit confirmation required before first cloud call to [{provider} ({model})].\n"
            f"Role: {role}, Content Classes: {content_classes}.\n"
            f"Run with human approval or create {ack_file}."
        )

    def record_acknowledgment(self, provider: str = "", model: str = "") -> None:
        """Save human acknowledgment to .swarm/cloud_ack.json."""
        self.swarm_dir.mkdir(parents=True, exist_ok=True)
        ack_file = self.swarm_dir / "cloud_ack.json"
        data = {
            "acknowledged_at": datetime.now(timezone.utc).isoformat(),
            "first_provider": provider,
            "first_model": model,
            "confirmed": True,
        }
        ack_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def filter_and_validate(
        self,
        role: str,
        provider: str,
        model: str,
        messages: list[ChatMessage],
        content_classes: list[str],
        file_names: list[str] | None = None,
        run_id: str = "",
    ) -> list[ChatMessage]:
        """Validate, redact, and audit an outbound cloud call."""
        # Local Ollama bypasses cloud outbound filter
        if provider.lower() == "ollama":
            return messages

        # 1. Offline check for any cloud provider
        if self.offline:
            raise OfflineModeActiveError(
                f"Cloud call blocked: --offline flag is active. Role '{role}' cannot call provider '{provider}'."
            )

        # 2. Human acknowledgment check
        self.check_cloud_acknowledgment(provider, model, role, content_classes)

        # 3. Deny list check on file names
        checked_files = file_names or []
        for fn in checked_files:
            for pat in self.hard_deny_patterns:
                if fnmatch.fnmatch(fn, pat) or fnmatch.fnmatch(os.path.basename(fn), pat):
                    raise SecurityPolicyViolationError(
                        f"Outbound payload contains forbidden file '{fn}' matching deny pattern '{pat}'"
                    )

        # 4. Content class validation per role
        role_allowed = self.allowed_classes_by_role.get(role, [])
        if "*" not in role_allowed:
            for cls in content_classes:
                if cls == "diffs" and not self.cloud_sees_code:
                    raise SecurityPolicyViolationError(
                        f"Role '{role}' attempted to send source diffs to cloud without --cloud-sees-code flag."
                    )
                if cls not in role_allowed:
                    raise SecurityPolicyViolationError(
                        f"Content class '{cls}' not permitted for cloud role '{role}' (allowed: {role_allowed})"
                    )

        # 5. Redaction and serialization
        redacted_messages: list[ChatMessage] = []
        total_bytes = 0

        for msg in messages:
            content = msg.content
            # Redact known sensitive patterns
            for regex in self.redaction_regexes:
                content = regex.sub("[REDACTED_SECRET]", content)

            # Redact any env var API keys currently loaded
            for env_key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"):
                val = os.getenv(env_key)
                if val and len(val) >= 8:
                    content = content.replace(val, "[REDACTED_API_KEY]")

            total_bytes += len(content.encode("utf-8"))
            redacted_messages.append(
                ChatMessage(
                    role=msg.role,
                    content=content,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )

        # 6. Size cap check
        if total_bytes > self.max_payload_bytes:
            raise PayloadSizeExceededError(
                f"Outbound payload size ({total_bytes} bytes) exceeds maximum permitted limit ({self.max_payload_bytes} bytes)."
            )

        # 7. Audit log
        self._write_audit_log(
            run_id=run_id,
            provider=provider,
            model=model,
            role=role,
            bytes_count=total_bytes,
            content_classes=content_classes,
            file_names=checked_files,
            messages=redacted_messages,
        )

        return redacted_messages

    def _write_audit_log(
        self,
        run_id: str,
        provider: str,
        model: str,
        role: str,
        bytes_count: int,
        content_classes: list[str],
        file_names: list[str],
        messages: list[ChatMessage],
    ) -> None:
        """Append record to outbound_audit.jsonl."""
        audit_dir = (self.swarm_dir / "runs" / run_id) if run_id else self.swarm_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / "outbound_audit.jsonl"

        # Compute payload hash
        full_text = "\n".join(m.content for m in messages)
        payload_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "role": role,
            "bytes": bytes_count,
            "content_classes": content_classes,
            "file_names": file_names,
            "payload_hash": payload_hash,
        }

        if self.log_payloads:
            entry["payload"] = [m.to_dict() for m in messages]

        with audit_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
