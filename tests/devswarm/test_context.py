"""
tests/devswarm/test_context.py — Tests for context management, 1500-token tool capping, and 70% compaction.
"""

from devswarm.agents.context import AgentContext, CHARS_PER_TOKEN


def test_context_basic_messages() -> None:
    ctx = AgentContext(num_ctx=16384)
    ctx.add_system("You are a backend specialist.")
    ctx.add_user("Implement safe_divide.")
    ctx.add_assistant("I will edit calc.py")

    messages = ctx.get_messages()
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_tool_output_capping_at_1500_tokens() -> None:
    ctx = AgentContext(num_ctx=16384)
    # 1500 tokens is 6000 chars. Produce a string of 10,000 characters.
    huge_output = "A" * 10000
    ctx.add_tool_result("read_file", huge_output)

    msg = ctx.messages[-1]
    assert msg.role == "tool"
    # Result must be capped at 6000 characters + truncation note
    assert len(msg.content) < 10000
    assert "Tool output truncated" in msg.content
    assert msg.content.startswith("A" * 6000)


def test_tool_output_secret_redaction() -> None:
    ctx = AgentContext(num_ctx=16384)
    secret_output = "Here is the key: sk-ant-api03-abcdef1234567890abcdef and password=SuperSecretPassword123"
    ctx.add_tool_result("run_command", secret_output)

    msg = ctx.messages[-1]
    assert "sk-ant-api03" not in msg.content
    assert "[REDACTED_SECRET]" in msg.content


def test_context_compaction_at_70_percent() -> None:
    # Use a small num_ctx to test compaction reliably
    # 1000 tokens * 0.70 = 700 tokens threshold (2800 chars)
    ctx = AgentContext(num_ctx=1000, compaction_ratio=0.70)
    ctx.add_system("SYSTEM: Specialist Engineer Prompt")
    ctx.add_user("USER: Task instructions")

    # Add several intermediate turns with tool outputs
    for i in range(5):
        ctx.add_assistant(f"Assistant step {i}")
        ctx.add_tool_result(f"tool_{i}", f"Result of step {i}: " + ("x" * 600))

    # Add recent working turns
    ctx.add_assistant("Final thoughts before completion")
    ctx.add_tool_result("tool_final", "Final verification output")

    # Initial token estimation should be high
    assert ctx.is_compaction_needed() is True
    initial_tokens = ctx.estimate_tokens()

    # Get messages triggers compaction
    compacted_messages = ctx.get_messages()

    # System and initial user must be preserved exactly
    assert compacted_messages[0]["content"] == "SYSTEM: Specialist Engineer Prompt"
    assert compacted_messages[1]["content"] == "USER: Task instructions"

    # Most recent turns must be preserved
    assert compacted_messages[-1]["content"] == "Final verification output"
    assert compacted_messages[-2]["content"] == "Final thoughts before completion"

    # Middle tool outputs must have been condensed
    assert ctx.compaction_count == 1
    assert ctx.estimate_tokens() < initial_tokens
    assert any("compacted to fit context" in m["content"] for m in compacted_messages)
