"""
devswarm.agents.loop — Specialist agent multi-turn tool execution loop.

Security & Execution Controls (16 GB profile):
  1. NUM_CTX 16384, num_predict 2048, LLM_CONCURRENCY 1.
  2. MAX_TURNS: 12 turns max per task.
  3. Loop Detection: aborts immediately if the same tool call with identical arguments
     repeats 3 times consecutively (LoopDetector).
  4. Kill Switch: checked before every single turn.
  5. Worktree Confinement: all file mutations strictly confined to worktree_dir.
  6. Narrow Write Scope: role and task write-scope enforced on every write_file / edit_file.
  7. Sandboxed Execution: run_command passes through Sandbox with network none and role command allowlist.
  8. Context Compaction: automatic compaction at 70% threshold.
  9. Secret Redaction: sensitive tokens and keys redacted from all tool results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devswarm.agents.context import AgentContext
from devswarm.agents.parser import extract_tool_calls, ParsedToolCall
from devswarm.agents.prompts import build_system_prompt
from devswarm.agents.roles import RoleSpec
from devswarm.llm.ollama_client import OllamaChatResponse, OllamaClient
from devswarm.platform.budget import SwarmBudget
from devswarm.platform.events import EventLog
from devswarm.platform.guards import LoopDetector, KillSwitch, KillSwitchTriggeredError, ThroughputGuard
from devswarm.platform.sandbox import Sandbox
from devswarm.providers.base import BaseProvider, ChatMessage, ToolDefinition
from devswarm.tools.command_tool import CommandTool
from devswarm.tools.fs_tools import FileSystemTools
from devswarm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionResult:
    """Outcome of a specialist agent task execution."""
    success: bool
    summary: str
    files_modified: list[str] = field(default_factory=list)
    error: str | None = None
    turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class AgentLoopError(Exception):
    """Raised when the agent loop encounters an unrecoverable condition."""


class LoopDetectedError(AgentLoopError):
    """Raised when 3 consecutive identical tool calls are detected."""


class MaxTurnsExceededError(AgentLoopError):
    """Raised when agent reaches turn limit without completing task."""


async def run_agent_loop(
    *,
    task_id: str,
    task_title: str,
    task_description: str,
    role_spec: RoleSpec,
    write_scope: list[str],
    acceptance_criteria: list[str],
    worktree_dir: Path | str,
    client: OllamaClient | BaseProvider,
    sandbox: Sandbox,
    budget: SwarmBudget,
    kill_switch: KillSwitch,
    event_log: EventLog,
    model: str | None = None,
    max_turns: int | None = None,
    num_ctx: int = 16384,
    num_predict: int = 2048,
    max_files: int = 2,
    throughput_guard: ThroughputGuard | None = None,
) -> TaskExecutionResult:
    """Execute specialist agent tool loop to fulfill a task."""
    effective_model = model or role_spec.model or "qwen3.5:9b"
    effective_max_turns = max_turns or role_spec.max_turns or 12
    worktree_path = Path(worktree_dir).resolve()

    logger.info(
        "Starting agent loop for task %s (Role: %s, Model: %s, Max turns: %d)",
        task_id,
        role_spec.name,
        effective_model,
        effective_max_turns,
    )

    # 1. Initialize Tools & Registry
    registry = ToolRegistry()
    fs_tools = FileSystemTools(
        worktree_root=worktree_path,
        write_scope=write_scope or role_spec.default_write_scope,
        max_output_chars=6000,
    )
    cmd_tool = CommandTool(
        sandbox=sandbox,
        worktree_dir=worktree_path,
        role_allowlist=role_spec.command_allowlist,
        timeout=30.0,
    )

    # Register standard filesystem tools
    registry.register(
        "read_file",
        "Read file contents or line range safely within worktree.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "start_line": {"type": "integer", "description": "1-based start line"},
                "end_line": {"type": "integer", "description": "1-based end line"},
            },
            "required": ["path"],
        },
        fs_tools.read_file,
    )
    registry.register(
        "write_file",
        "Write full content to a file inside the task write scope (max 2 files).",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within write scope"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
        fs_tools.write_file,
    )
    registry.register(
        "edit_file",
        "Perform exact unique string replacement in a file inside write scope.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within write scope"},
                "old_str": {"type": "string", "description": "Exact text to find and replace"},
                "new_str": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_str", "new_str"],
        },
        fs_tools.edit_file,
    )
    registry.register(
        "list_dir",
        "List files and subdirectories in worktree.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path (default: .)"},
            },
        },
        fs_tools.list_dir,
    )
    registry.register(
        "search",
        "Search text occurrences in files across the worktree.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text substring to search"},
                "path": {"type": "string", "description": "Subdirectory to search (default: .)"},
            },
            "required": ["pattern"],
        },
        fs_tools.search,
    )

    # Register sandboxed command tool if role allows run_command
    if "run_command" in role_spec.allowed_tools:
        registry.register(
            "run_command",
            "Execute a sandboxed shell command (e.g. pytest, python -m py_compile).",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run in sandbox"},
                },
                "required": ["command"],
            },
            cmd_tool.execute,
        )

    # Terminal state flags
    is_finished = False
    finish_summary = ""
    is_failed = False
    fail_reason = ""
    modified_files: set[str] = set()

    def finish_task_handler(summary: str, files_modified: list[str] | None = None) -> str:
        nonlocal is_finished, finish_summary
        is_finished = True
        finish_summary = summary
        if files_modified:
            modified_files.update(files_modified)
        return "Task marked as finished. Verifying quality gates."

    def fail_task_handler(reason: str) -> str:
        nonlocal is_failed, fail_reason
        is_failed = True
        fail_reason = reason
        return f"Task marked as failed: {reason}"

    registry.register(
        "finish_task",
        "Signal that the task is complete and all acceptance criteria are met.",
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of changes made"},
                "files_modified": {"type": "array", "items": {"type": "string"}, "description": "List of files modified"},
            },
            "required": ["summary"],
        },
        finish_task_handler,
    )

    registry.register(
        "fail_task",
        "Signal that the task is blocked, impossible, or failed constraints.",
        {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Detailed explanation of why task failed"},
            },
            "required": ["reason"],
        },
        fail_task_handler,
    )

    # 2. Setup Context & Loop Guard
    context = AgentContext(num_ctx=num_ctx)
    system_prompt = build_system_prompt(
        role=role_spec.name,
        task_id=task_id,
        task_title=task_title,
        task_description=task_description,
        write_scope=write_scope or role_spec.default_write_scope,
        acceptance_criteria=acceptance_criteria,
        max_files=max_files,
    )
    context.add_system(system_prompt)

    user_instructions = (
        f"Subtask: [{task_id}] {task_title}\n\n"
        f"Description:\n{task_description}\n\n"
        f"Acceptance Criteria:\n" + "\n".join(f"- {c}" for c in acceptance_criteria) + "\n\n"
        "Begin work. Read necessary files, implement code or tests, verify your changes, and call finish_task."
    )
    context.add_user(user_instructions)

    loop_detector = LoopDetector(threshold=3)
    tools_schema = registry.get_tools_for_role()

    total_in = 0
    total_out = 0

    # 3. Execution Turns Loop
    for turn in range(1, effective_max_turns + 1):
        # Check kill switch before every turn
        kill_switch.check()

        # Turn budget reminders on final turns
        if turn == effective_max_turns:
            context.add_user(
                "⚠️ FINAL TURN WARNING: This is your final turn. "
                "If your changes are ready, you MUST call the `finish_task` tool now."
            )
        elif turn == effective_max_turns - 2:
            context.add_user(
                "Turn budget reminder: 2 turns remaining. Finalize your implementation and call `finish_task`."
            )

        event_log.emit("agent_turn_start", task_id=task_id, role=role_spec.name, args={"turn": turn})
        logger.info("Task %s [%s] -> Turn %d/%d", task_id, role_spec.name, turn, effective_max_turns)

        # Chat with LLM
        messages = context.get_messages()
        if isinstance(client, BaseProvider):
            chat_messages = [ChatMessage.from_dict(m) for m in messages]
            tool_defs = registry.get_tool_definitions_for_role() if "run_command" in role_spec.allowed_tools or role_spec.name in ("backend", "tester", "qa", "architect") else None
            p_resp = await client.chat(
                model=effective_model,
                messages=chat_messages,
                tools=tool_defs,
                temperature=0.2,
                max_tokens=num_predict,
            )
            tok_in = p_resp.usage.input_tokens
            tok_out = p_resp.usage.output_tokens
            tps = (tok_out / p_resp.latency_s) if p_resp.latency_s > 0 else 0.0
            tool_calls_raw = [{"function": {"name": tc.name, "arguments": tc.arguments}, "id": tc.id} for tc in p_resp.tool_calls]
            resp = OllamaChatResponse(
                model=effective_model,
                content=p_resp.content,
                tool_calls=tool_calls_raw,
                prompt_eval_count=tok_in,
                eval_count=tok_out,
                tokens_per_sec=tps,
                time_to_first_token_s=p_resp.latency_s,
                load_duration_s=0.0,
            )
        else:
            resp = await client.chat(
                model=effective_model,
                messages=messages,
                tools=tools_schema,
                options={"num_ctx": num_ctx, "temperature": 0.2, "num_predict": num_predict},
                keep_alive="-1",
            )

        total_in += resp.prompt_eval_count
        total_out += resp.eval_count

        # Telemetry recording
        budget.record_call(
            model=effective_model,
            input_tokens=resp.prompt_eval_count,
            output_tokens=resp.eval_count,
            tokens_per_sec=resp.tokens_per_sec,
            time_to_first_token_s=resp.time_to_first_token_s,
            load_duration_s=resp.load_duration_s,
            context_size_used=resp.prompt_eval_count + resp.eval_count,
            task_id=task_id,
            role=role_spec.name,
        )

        if throughput_guard is not None and resp.tokens_per_sec > 0:
            throughput_guard.record_call(resp.tokens_per_sec)

        # Extract tool calls tolerantly
        tool_calls = extract_tool_calls(
            response_tool_calls=resp.tool_calls,
            content=resp.content,
        )

        # Add assistant message to history
        formatted_tool_calls = [
            {
                "id": t.tool_call_id or f"call_{turn}_{i}",
                "name": t.name,
                "arguments": t.arguments,
                "function": {"name": t.name, "arguments": t.arguments},
            }
            for i, t in enumerate(tool_calls)
        ] if tool_calls else None

        context.add_assistant(
            content=resp.content,
            tool_calls=formatted_tool_calls,
        )

        if not tool_calls:
            logger.info("Agent provided commentary without tool calls on turn %d", turn)

            # Discover all modified files: explicit tool calls, git status changes
            actual_modified = set(modified_files)
            try:
                p_status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(worktree_path),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in p_status.stdout.splitlines():
                    line = line.strip()
                    if line:
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            actual_modified.add(parts[1].strip("\"' "))
            except Exception:
                pass

            # If the model provides commentary (or reaches final turn) and files were modified,
            # treat it as completion and delegate verification to the deterministic quality gates!
            if actual_modified:
                logger.info(
                    "Agent provided commentary on turn %d with modified files %s; proceeding to deterministic quality gates.",
                    turn,
                    actual_modified,
                )
                return TaskExecutionResult(
                    success=True,
                    summary=resp.content.strip() or "Task completed; verifying quality gates.",
                    files_modified=sorted(actual_modified),
                    turns=turn,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

            if turn < effective_max_turns:
                context.add_user(
                    "CRITICAL: You must invoke a tool. Do NOT reply with text explanations or commentary alone. "
                    "Call an appropriate tool (e.g. read_file, edit_file, write_file, run_command), "
                    "or call `finish_task` with your summary if all acceptance criteria are met."
                )
                continue
            else:
                return TaskExecutionResult(
                    success=False,
                    summary="Exceeded max turns without calling finish_task",
                    error="Max turns reached without terminal tool call",
                    turns=turn,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

        # Execute extracted tool calls
        for i, tc in enumerate(tool_calls):
            call_id = tc.tool_call_id or f"call_{turn}_{i}"
            # Check loop detector for runaway repeated tool calls
            loop_detector.record(tc.name, tc.arguments)
            if loop_detector.is_looping():
                logger.error("Loop detector triggered: 3 consecutive calls to %s with %s", tc.name, tc.arguments)
                event_log.emit("loop_detected", task_id=task_id, role=role_spec.name, args={"tool": tc.name, "arguments": tc.arguments})
                return TaskExecutionResult(
                    success=False,
                    summary="Task aborted due to repetitive tool call loop",
                    error=f"Loop detected: 3 identical consecutive calls to {tc.name}",
                    turns=turn,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

            # Track modified files
            if tc.name in ("write_file", "edit_file") and "path" in tc.arguments:
                modified_files.add(str(tc.arguments["path"]))

            event_log.emit("tool_call_start", task_id=task_id, role=role_spec.name, args={"tool": tc.name})
            tool_output = await registry.execute(tc.name, tc.arguments)
            event_log.emit("tool_call_complete", task_id=task_id, role=role_spec.name, args={"tool": tc.name})

            # Add tool result to context (auto secret-redacted & capped at 1500 tokens)
            context.add_tool_result(name=tc.name, result=tool_output, tool_call_id=call_id)

            if is_finished:
                logger.info("Task %s completed successfully at turn %d", task_id, turn)
                return TaskExecutionResult(
                    success=True,
                    summary=finish_summary or "Task finished successfully.",
                    files_modified=sorted(modified_files),
                    turns=turn,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

            if is_failed:
                logger.warning("Task %s declared failed at turn %d: %s", task_id, turn, fail_reason)
                return TaskExecutionResult(
                    success=False,
                    summary="Task declared failure.",
                    files_modified=sorted(modified_files),
                    error=fail_reason or "Declared failure by agent",
                    turns=turn,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

    # Discover any modifications on disk via git status before final evaluation
    try:
        p_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in p_status.stdout.splitlines():
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    modified_files.add(parts[1].strip("\"' "))
    except Exception:
        pass

    if modified_files and not is_failed:
        logger.info(
            "Task %s reached turn limit but modified files (%s); proceeding to deterministic quality gates",
            task_id, sorted(modified_files),
        )
        return TaskExecutionResult(
            success=True,
            summary=f"Task reached turn limit with modified files ({', '.join(sorted(modified_files))}). Submitting to quality gates.",
            files_modified=sorted(modified_files),
            turns=effective_max_turns,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    return TaskExecutionResult(
        success=False,
        summary="Exceeded max turns",
        error=f"Exceeded max turns limit ({effective_max_turns})",
        turns=effective_max_turns,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
    )
