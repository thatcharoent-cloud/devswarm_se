"""
tests.devswarm.test_outbound_filter — Unit tests for the outbound security filter.

Verifies:
  - First-call confirmation requirement (.swarm/cloud_ack.json)
  - Offline mode blocking cloud calls
  - Hard deny list patterns (.env, keys, certs, .git, .swarm)
  - Role-based content class allowlists (and --cloud-sees-code for diffs)
  - Automatic redaction of API keys and high-entropy secrets
  - Payload size cap enforcement
  - Outbound audit log emission
"""

import json
from pathlib import Path
import pytest

from devswarm.platform.outbound_filter import (
    CloudConfirmationRequiredError,
    OfflineModeActiveError,
    OutboundFilter,
    PayloadSizeExceededError,
    SecurityPolicyViolationError,
)
from devswarm.providers.base import ChatMessage


def test_first_call_cloud_acknowledgment(tmp_path):
    swarm_dir = tmp_path / ".swarm"
    filter_ = OutboundFilter(swarm_dir=swarm_dir)

    # 1. First call without ack raises error
    with pytest.raises(CloudConfirmationRequiredError):
        filter_.filter_and_validate(
            role="pm",
            provider="anthropic",
            model="claude-3-7-sonnet",
            messages=[ChatMessage(role="user", content="Draft spec")],
            content_classes=["goal"],
        )

    # 2. Provide callback that confirms
    called = []
    def confirm_cb(provider, model, role, classes):
        called.append((provider, model, role))
        return True

    filter_cb = OutboundFilter(swarm_dir=swarm_dir, confirmation_callback=confirm_cb)
    msgs = filter_cb.filter_and_validate(
        role="pm",
        provider="anthropic",
        model="claude-3-7-sonnet",
        messages=[ChatMessage(role="user", content="Draft spec")],
        content_classes=["goal"],
    )
    assert len(msgs) == 1
    assert len(called) == 1
    assert (swarm_dir / "cloud_ack.json").exists()

    # 3. Subsequent calls now pass automatically
    msgs2 = filter_.filter_and_validate(
        role="pm",
        provider="anthropic",
        model="claude-3-7-sonnet",
        messages=[ChatMessage(role="user", content="Draft spec")],
        content_classes=["goal"],
    )
    assert len(msgs2) == 1


def test_offline_mode_blocking(tmp_path):
    swarm_dir = tmp_path / ".swarm"
    filter_ = OutboundFilter(swarm_dir=swarm_dir, offline=True)
    filter_.record_acknowledgment()

    # Cloud call fails under offline mode
    with pytest.raises(OfflineModeActiveError):
        filter_.filter_and_validate(
            role="pm",
            provider="anthropic",
            model="claude-3-7-sonnet",
            messages=[ChatMessage(role="user", content="Draft spec")],
            content_classes=["goal"],
        )

    # Ollama call succeeds even when offline flag is on
    msgs = filter_.filter_and_validate(
        role="backend",
        provider="ollama",
        model="qwen2.5-coder:7b",
        messages=[ChatMessage(role="user", content="Code feature")],
        content_classes=["*"],
    )
    assert len(msgs) == 1


def test_hard_deny_patterns(tmp_path):
    swarm_dir = tmp_path / ".swarm"
    filter_ = OutboundFilter(swarm_dir=swarm_dir)
    filter_.record_acknowledgment()

    # Including .env file
    with pytest.raises(SecurityPolicyViolationError) as exc_info:
        filter_.filter_and_validate(
            role="architect",
            provider="openai",
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Analyze config")],
            content_classes=["architecture"],
            file_names=[".env.production"],
        )
    assert ".env" in str(exc_info.value)

    # Including private key
    with pytest.raises(SecurityPolicyViolationError):
        filter_.filter_and_validate(
            role="architect",
            provider="openai",
            model="gpt-4o",
            messages=[ChatMessage(role="user", content="Analyze keys")],
            content_classes=["architecture"],
            file_names=["server.key"],
        )


def test_role_content_classes_and_cloud_sees_code(tmp_path):
    swarm_dir = tmp_path / ".swarm"
    filter_ = OutboundFilter(swarm_dir=swarm_dir, cloud_sees_code=False)
    filter_.record_acknowledgment()

    # PM allowed classes: goal, spec, repo_map, user_answers
    # Attempting to send diffs without --cloud-sees-code fails
    with pytest.raises(SecurityPolicyViolationError) as exc_info:
        filter_.filter_and_validate(
            role="pm",
            provider="anthropic",
            model="claude-3-7-sonnet",
            messages=[ChatMessage(role="user", content="Review diffs")],
            content_classes=["diffs"],
        )
    assert "--cloud-sees-code" in str(exc_info.value)

    # Attempting to send unallowed class fails
    with pytest.raises(SecurityPolicyViolationError):
        filter_.filter_and_validate(
            role="pm",
            provider="anthropic",
            model="claude-3-7-sonnet",
            messages=[ChatMessage(role="user", content="Gate logs")],
            content_classes=["gate_logs"],
        )

    # Enabling cloud_sees_code allows diffs for reviewer
    filter_allowed = OutboundFilter(swarm_dir=swarm_dir, cloud_sees_code=True)
    msgs = filter_allowed.filter_and_validate(
        role="reviewer",
        provider="anthropic",
        model="claude-3-7-sonnet",
        messages=[ChatMessage(role="user", content="Review diffs")],
        content_classes=["diffs"],
    )
    assert len(msgs) == 1


def test_secret_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live-real-secret-123456789")
    swarm_dir = tmp_path / ".swarm"
    filter_ = OutboundFilter(swarm_dir=swarm_dir, log_payloads=True)
    filter_.record_acknowledgment()

    secret_prompt = (
        "Here is my OpenAI key: sk-abcdefghijklmnopqrstuvwxyz1234567890\n"
        "Here is my Anthropic key: sk-ant-live-real-secret-123456789\n"
        "And Gemini: AIzaSyD9ABCDEF1234567890123456789012345\n"
        "Please build the feature."
    )

    msgs = filter_.filter_and_validate(
        role="pm",
        provider="anthropic",
        model="claude-3-7-sonnet",
        messages=[ChatMessage(role="user", content=secret_prompt)],
        content_classes=["goal"],
        run_id="run_test_01",
    )

    clean_content = msgs[0].content
    assert "sk-abcdef" not in clean_content
    assert "sk-ant-live-real" not in clean_content
    assert "AIzaSyD9" not in clean_content
    assert "[REDACTED" in clean_content

    # Verify audit log exists and also has redacted contents
    audit_file = swarm_dir / "runs" / "run_test_01" / "outbound_audit.jsonl"
    assert audit_file.exists()
    audit_content = audit_file.read_text(encoding="utf-8")
    assert "sk-ant-live-real" not in audit_content


def test_payload_size_limit(tmp_path):
    swarm_dir = tmp_path / ".swarm"
    filter_ = OutboundFilter(swarm_dir=swarm_dir)
    filter_.record_acknowledgment()
    filter_.max_payload_bytes = 1000  # Set low limit for test

    giant_prompt = "x" * 1500
    with pytest.raises(PayloadSizeExceededError):
        filter_.filter_and_validate(
            role="pm",
            provider="anthropic",
            model="claude-3-7-sonnet",
            messages=[ChatMessage(role="user", content=giant_prompt)],
            content_classes=["goal"],
        )

