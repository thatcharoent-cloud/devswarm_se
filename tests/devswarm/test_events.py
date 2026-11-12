"""
tests/devswarm/test_events.py — Unit tests for event logging, secret redaction, and history reconstruction.
"""

import json
from pathlib import Path
from devswarm.platform.events import EventLog, redact_secrets, truncate_payload, compute_prompt_hash


def test_secret_redaction():
    # OpenAI key
    raw = "Here is my secret: sk-abcdef1234567890abcdef1234567890 and password: 'supersecret123'"
    redacted = redact_secrets(raw)
    assert "sk-" not in redacted or "[REDACTED_SECRET]" in redacted
    assert "supersecret123" not in redacted

    # Dictionary with secret key
    d = {"api_key": "secret_token_value_999", "normal_field": "hello world"}
    redacted_d = redact_secrets(d)
    assert redacted_d["api_key"] == "[REDACTED_SECRET]"
    assert redacted_d["normal_field"] == "hello world"


def test_payload_truncation():
    long_str = "x" * 5000
    truncated = truncate_payload(long_str, max_chars=100)
    assert len(truncated) < 200
    assert "[TRUNCATED" in truncated


def test_prompt_hash():
    p1 = "System prompt version 1"
    p2 = "System prompt version 2"
    h1 = compute_prompt_hash(p1)
    h2 = compute_prompt_hash(p2)
    assert len(h1) == 16
    assert h1 != h2
    assert compute_prompt_hash(p1) == h1


def test_event_log_append_and_reconstruct(tmp_path):
    log = EventLog(run_dir=tmp_path, run_id="run-101")

    # Emit run start
    log.emit("run_start", args={"goal": "Fix bug in calc.py"})

    # Emit task 1 events
    log.emit("task_start", task_id="task-1", role="backend")
    log.emit("tool_call", task_id="task-1", role="backend", args={"tool": "read_file", "path": "calc.py"})
    log.emit("task_done", task_id="task-1", role="backend")

    # Emit task 2 event
    log.emit("task_start", task_id="task-2", role="tester")

    events = log.read_all()
    assert len(events) == 5

    # Check task 1 reconstruction
    history = log.reconstruct_task_history("task-1")
    assert len(history) == 3
    assert [e["type"] for e in history] == ["task_start", "tool_call", "task_done"]
    assert all(e["task_id"] == "task-1" for e in history)

