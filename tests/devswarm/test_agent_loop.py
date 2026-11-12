"""
tests/devswarm/test_agent_loop.py — Tests for multi-turn agent tool execution loop.
"""

from pathlib import Path
import pytest

from devswarm.agents.loop import run_agent_loop, TaskExecutionResult
from devswarm.agents.roles import RoleSpec
from devswarm.llm.mock_llm import MockLLM
from devswarm.platform.budget import SwarmBudget
from devswarm.platform.events import EventLog
from devswarm.platform.guards import KillSwitch, KillSwitchTriggeredError
from devswarm.platform.sandbox import MockSandbox


@pytest.fixture
def test_env(tmp_path: Path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    run_dir = tmp_path / "run_01"
    run_dir.mkdir()

    client = MockLLM()
    sandbox = MockSandbox()
    budget = SwarmBudget(max_run_minutes=10.0, max_task_minutes=5.0)
    stop_file = tmp_path / "STOP"
    kill_switch = KillSwitch(stop_file=stop_file)
    event_log = EventLog(run_dir=run_dir, run_id="test_run")

    role_spec = RoleSpec(
        name="backend",
        enabled=True,
        allowed_tools=["read_file", "write_file", "edit_file", "list_dir", "search", "run_command"],
        default_write_scope=["calc.py"],
        command_allowlist=["pytest", "python"],
        max_turns=6,
    )

    return {
        "worktree": worktree,
        "run_dir": run_dir,
        "client": client,
        "sandbox": sandbox,
        "budget": budget,
        "kill_switch": kill_switch,
        "event_log": event_log,
        "role_spec": role_spec,
    }


@pytest.mark.asyncio
async def test_agent_loop_successful_execution(test_env) -> None:
    client: MockLLM = test_env["client"]

    # Scripted 2-turn execution:
    # Turn 1: edit_file
    # Turn 2: finish_task
    client.set_scripted_responses([
        {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "edit_file",
                    "arguments": {
                        "path": "calc.py",
                        "old_str": "return a + b",
                        "new_str": "return a + b  # verified",
                    },
                }
            }],
        },
        {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "finish_task",
                    "arguments": {
                        "summary": "Added inline verification comment",
                        "files_modified": ["calc.py"],
                    },
                }
            }],
        },
    ])

    res = await run_agent_loop(
        task_id="task-1",
        task_title="Update calc add logic",
        task_description="Add verification comment",
        role_spec=test_env["role_spec"],
        write_scope=["calc.py"],
        acceptance_criteria=["Comment present in calc.py"],
        worktree_dir=test_env["worktree"],
        client=client,
        sandbox=test_env["sandbox"],
        budget=test_env["budget"],
        kill_switch=test_env["kill_switch"],
        event_log=test_env["event_log"],
        max_turns=5,
    )

    assert res.success is True
    assert res.turns == 2
    assert "calc.py" in res.files_modified
    assert "verified" in (test_env["worktree"] / "calc.py").read_text(encoding="utf-8")
    assert test_env["budget"].summary()["total_llm_calls"] >= 2


@pytest.mark.asyncio
async def test_agent_loop_loop_detector_triggers(test_env) -> None:
    client: MockLLM = test_env["client"]

    # Script 3 identical tool calls consecutively
    identical_call = {
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "read_file",
                "arguments": {"path": "calc.py"},
            }
        }],
    }
    client.set_scripted_responses([identical_call, identical_call, identical_call])

    res = await run_agent_loop(
        task_id="task-loop",
        task_title="Looping task",
        task_description="Test loop guard",
        role_spec=test_env["role_spec"],
        write_scope=["calc.py"],
        acceptance_criteria=["Should abort"],
        worktree_dir=test_env["worktree"],
        client=client,
        sandbox=test_env["sandbox"],
        budget=test_env["budget"],
        kill_switch=test_env["kill_switch"],
        event_log=test_env["event_log"],
        max_turns=6,
    )

    assert res.success is False
    assert "Loop detected" in str(res.error)


@pytest.mark.asyncio
async def test_agent_loop_write_scope_rejection(test_env) -> None:
    client: MockLLM = test_env["client"]

    # Attempt to write outside write_scope (write_scope is calc.py, target is secret.py)
    client.set_scripted_responses([
        {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": {
                        "path": "secret.py",
                        "content": "secret = 123",
                    },
                }
            }],
        },
        {
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "finish_task",
                    "arguments": {"summary": "Done"},
                }
            }],
        },
    ])

    res = await run_agent_loop(
        task_id="task-scope",
        task_title="Scope test",
        task_description="Test scope protection",
        role_spec=test_env["role_spec"],
        write_scope=["calc.py"],
        acceptance_criteria=["Must not write secret.py"],
        worktree_dir=test_env["worktree"],
        client=client,
        sandbox=test_env["sandbox"],
        budget=test_env["budget"],
        kill_switch=test_env["kill_switch"],
        event_log=test_env["event_log"],
        max_turns=4,
    )

    # secret.py must NOT exist
    assert not (test_env["worktree"] / "secret.py").exists()


@pytest.mark.asyncio
async def test_agent_loop_kill_switch_triggers(test_env) -> None:
    # Trigger kill switch before or during loop
    test_env["kill_switch"].trigger("Operator stop")

    with pytest.raises(KillSwitchTriggeredError):
        await run_agent_loop(
            task_id="task-kill",
            task_title="Killed task",
            task_description="Should abort on kill switch",
            role_spec=test_env["role_spec"],
            write_scope=["calc.py"],
            acceptance_criteria=["None"],
            worktree_dir=test_env["worktree"],
            client=test_env["client"],
            sandbox=test_env["sandbox"],
            budget=test_env["budget"],
            kill_switch=test_env["kill_switch"],
            event_log=test_env["event_log"],
            max_turns=4,
        )


@pytest.mark.asyncio
async def test_agent_loop_max_turns_exceeded(test_env) -> None:
    client: MockLLM = test_env["client"]

    # Always return commentary with no terminal call
    client.set_scripted_responses([
        "Thinking turn 1...",
        "Thinking turn 2...",
        "Thinking turn 3...",
    ])

    res = await run_agent_loop(
        task_id="task-turns",
        task_title="Endless thinking task",
        task_description="Hit max turns",
        role_spec=test_env["role_spec"],
        write_scope=["calc.py"],
        acceptance_criteria=["None"],
        worktree_dir=test_env["worktree"],
        client=client,
        sandbox=test_env["sandbox"],
        budget=test_env["budget"],
        kill_switch=test_env["kill_switch"],
        event_log=test_env["event_log"],
        max_turns=3,
    )

    assert res.success is False
    assert "max turns" in str(res.error).lower() or "max turns" in str(res.summary).lower()


@pytest.mark.asyncio
async def test_agent_loop_with_base_provider(test_env) -> None:
    from devswarm.providers.mock_provider import MockProvider
    from devswarm.providers.base import ToolCall

    provider = MockProvider(name="siliconflow")
    provider.add_response(
        content="",
        tool_calls=[ToolCall(id="call_1", name="finish_task", arguments={"summary": "Task complete via cloud provider", "files_modified": ["calc.py"]})],
    )

    res = await run_agent_loop(
        task_id="task-cloud-1",
        task_title="Cloud agent task execution",
        task_description="Execute using BaseProvider",
        role_spec=test_env["role_spec"],
        write_scope=["calc.py"],
        acceptance_criteria=["Task complete"],
        worktree_dir=test_env["worktree"],
        client=provider,
        sandbox=test_env["sandbox"],
        budget=test_env["budget"],
        kill_switch=test_env["kill_switch"],
        event_log=test_env["event_log"],
        model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
    )

    assert res.success is True
    assert "complete via cloud provider" in res.summary
