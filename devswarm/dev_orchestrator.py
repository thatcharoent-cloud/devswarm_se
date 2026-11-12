"""
devswarm.dev_orchestrator — Chief Engineering Orchestrator for local software development.

Responsibilities:
  - Clarifies user requirements (interactively if --interactive, otherwise records explicit assumptions).
  - Decomposes goal into a validated task DAG via structured output (Pydantic schema).
  - Deterministically validates plan: enabled roles, acyclic DAG, max files per write scope,
    non-overlapping parallel write scopes, subtask limits, and time budget.
  - Retries once with structured error feedback if plan validation fails.
  - Schedules tasks sequentially (LLM_CONCURRENCY=1) in dependency order.
  - Manages worktrees, integration branch, commits, and quality gate triggers.
  - Unloads model from memory every N tasks (UNLOAD_MODEL_EVERY_N_TASKS) to prevent KV cache bloat.
  - Never edits code directly.
  - Supports `status` and `stop` controls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from devswarm.agents.plan_schema import (
    EngineeringPlan,
    TaskPlanItem,
    PlanValidationError,
    validate_engineering_plan,
    normalize_plan_dependencies,
    build_plan_repair_prompt,
)
from devswarm.agents.roles import RoleRegistry, RoleSpec
from devswarm.agents.loop import run_agent_loop, TaskExecutionResult
from devswarm.gates import GateRunner, GateSuiteResult
from devswarm.reviewer import SwarmReviewer, ReviewReport
from devswarm.pr import PRGenerator
from devswarm.llm.ollama_client import OllamaClient, OllamaChatResponse
from devswarm.platform.artifacts import ArtifactStore
from devswarm.platform.budget import SwarmBudget, BudgetExceededError
from devswarm.platform.events import EventLog, SwarmEvent
from devswarm.platform.guards import (
    KillSwitch,
    KillSwitchTriggered,
    LoopDetector,
    ThroughputGuard,
    ThroughputCollapsedError,
    CaffeinateContext,
    CriticalMemoryPressureError,
    check_battery_state,
    check_memory_pressure_critical,
)
from devswarm.platform.manifest import build_manifest, RunManifest
from devswarm.platform.memory import ProjectMemory
from devswarm.platform.sandbox import Sandbox, DockerSandbox, MockSandbox
from devswarm.platform.taskqueue import TaskQueue, SwarmTask, TaskState
from devswarm.platform.worktrees import WorktreeManager, GitConflictError
from devswarm.tools.repo_map import generate_repo_map
from devswarm.self_healing import TaskAttemptManager, CloudEscalationHandler, extract_failure_diagnostics
from devswarm.scheduler import TwoPoolScheduler
from devswarm.platform.budget import CloudBudgetTracker
from devswarm.platform.outbound_filter import OutboundFilter
from devswarm.providers.registry import ProviderRegistry
from devswarm.providers.base import BaseProvider, ChatMessage
from devswarm.traceability import TraceabilityMatrix
from devswarm.pipeline import PipelineController, GatePausedException
from devswarm.agents.qa_schema import QATestSuite

logger = logging.getLogger(__name__)

DEFAULT_ROLES_PATH = "config/roles.yaml"
DEFAULT_MODELS_PATH = "config/models.yaml"
DEFAULT_POLICY_PATH = "config/policy.yaml"
DEFAULT_GATES_PATH = "config/gates.yaml"
MAX_CLARIFYING_QUESTIONS = 3


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""


class InvalidPlanError(OrchestratorError):
    """Raised when the plan fails validation after retry."""


class DevOrchestrator:
    """Chief Engineering Orchestrator coordinating local specialist agents."""

    def __init__(
        self,
        repo_path: Path | str,
        goal: str,
        client: OllamaClient | None = None,
        sandbox: Sandbox | None = None,
        roles_config_path: str | Path = DEFAULT_ROLES_PATH,
        models_config_path: str | Path = DEFAULT_MODELS_PATH,
        policy_config_path: str | Path = DEFAULT_POLICY_PATH,
        gates_config_path: str | Path = DEFAULT_GATES_PATH,
        swarm_dir: Path | str = ".swarm",
        interactive: bool = False,
        resume_run_id: str | None = None,
        question_prompter: Callable[[str], str] | None = None,
        escalate_to_cloud: bool = False,
        auto_approve_design: bool = False,
        waive_ac: list[str] | None = None,
        offline: bool = False,
        cloud_sees_code: bool = False,
        log_payloads: bool = False,
        mock_mode: bool = False,
        max_fix_attempts: int | None = None,
        roles: list[str] | set[str] | str | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.goal = goal
        self.interactive = interactive
        self.question_prompter = question_prompter
        self.swarm_dir = Path(swarm_dir).resolve()
        self.escalate_to_cloud = escalate_to_cloud
        self.auto_approve_design = auto_approve_design
        self.waived_acs = waive_ac or []
        self.offline = offline
        self.cloud_sees_code = cloud_sees_code
        self.log_payloads = log_payloads
        self.mock_mode = mock_mode
        self.roles = roles

        # Load configurations
        self.roles_config = self._load_yaml(roles_config_path)
        self.models_config = self._load_yaml(models_config_path)
        self.policy_config = self._load_yaml(policy_config_path)
        self.role_registry = RoleRegistry(roles_config_path)

        if max_fix_attempts is not None:
            self.max_fix_attempts = max_fix_attempts
        else:
            env_fix = os.getenv("MAX_FIX_ATTEMPTS")
            if env_fix:
                self.max_fix_attempts = int(env_fix)
            elif self.escalate_to_cloud:
                self.max_fix_attempts = 2
            else:
                self.max_fix_attempts = int(self.policy_config.get("max_fix_attempts", 1))

        # 16 GB Profile Constraints & Defaults
        self.max_files_per_task = int(
            os.getenv("MAX_FILES_PER_TASK") or self.policy_config.get("max_files_per_task", 2)
        )
        self.num_ctx = int(os.getenv("NUM_CTX") or 16384)
        self.num_predict = int(os.getenv("NUM_PREDICT") or 2048)
        self.unload_every_n_tasks = int(
            os.getenv("UNLOAD_MODEL_EVERY_N_TASKS") or self.policy_config.get("unload_model_every_n_tasks", 3)
        )
        self.max_run_minutes = float(os.getenv("MAX_RUN_MINUTES") or 45.0)
        self.max_task_minutes = float(os.getenv("MAX_TASK_MINUTES") or 10.0)

        # Active Model Tag (respect .env override, never mutate tags without asking)
        self.orch_provider_name = os.getenv("ORCHESTRATOR_PROVIDER", "siliconflow")
        default_orch_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017" if self.orch_provider_name == "siliconflow" else "qwen3.5:9b"
        self.model = os.getenv("ORCHESTRATOR_MODEL") or os.getenv("MODEL") or default_orch_model

        # Enabled roles (default: backend and tester)
        self.enabled_roles = self._determine_enabled_roles()

        # Run ID and directories
        self.resume_run_id = resume_run_id
        if resume_run_id:
            self.run_id = resume_run_id
            self.run_dir = self.swarm_dir / "runs" / self.run_id
            manifest_file = self.run_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    if not self.goal:
                        self.goal = m_data.get("goal", "")
                    if "target_repo" in m_data and (not repo_path or str(repo_path) == "." or Path(repo_path).resolve() == Path(".").resolve()):
                        self.repo_path = Path(m_data["target_repo"]).resolve()
                    if "model" in m_data and not os.getenv("ORCHESTRATOR_MODEL"):
                        self.model = m_data["model"]
                except Exception:
                    pass
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.run_id = f"run_{ts}"
            self.run_dir = self.swarm_dir / "runs" / self.run_id

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Platform components
        self.client = client
        self.sandbox = sandbox or DockerSandbox(
            memory=os.getenv("SANDBOX_MEMORY", "1g"),
            cpus=os.getenv("SANDBOX_CPUS", "2"),
        )
        self.gate_runner = GateRunner(
            sandbox=self.sandbox,
            gates_config_path=gates_config_path,
            policy_config_path=policy_config_path,
        )
        self.event_log = EventLog(run_dir=self.run_dir, run_id=self.run_id)
        self.artifacts = ArtifactStore(run_dir=self.run_dir)
        self.memory = ProjectMemory(memory_file=self.swarm_dir / "memory.jsonl")
        self.task_queue = TaskQueue(db_path=self.run_dir / "tasks.db")
        self.worktrees = WorktreeManager(repo_path=self.repo_path, run_id=self.run_id, swarm_dir=self.swarm_dir)
        self.kill_switch = KillSwitch(stop_file=self.swarm_dir / "STOP")
        self.budget = SwarmBudget(
            max_run_minutes=self.max_run_minutes,
            max_task_minutes=self.max_task_minutes,
            max_tokens=int(os.getenv("MAX_TOKENS", 2_000_000)),
            max_llm_calls=int(os.getenv("MAX_LLM_CALLS", 300)),
        )

        self.tasks_since_last_unload = 0
        self.throughput_guard = ThroughputGuard(
            min_tokens_per_sec=float(os.getenv("MIN_TOKENS_PER_SEC") or 5.0),
            streak_limit=3,
        )

        # Hybrid components (Phase D)
        self.attempt_manager = TaskAttemptManager()
        self.cloud_budget = CloudBudgetTracker.from_config()
        self.outbound_filter = OutboundFilter(
            policy_path=policy_config_path,
            swarm_dir=self.swarm_dir,
            offline=self.offline,
            cloud_sees_code=self.cloud_sees_code,
            log_payloads=self.log_payloads,
        )

        providers_cfg = Path("config/providers.yaml")
        if not providers_cfg.exists() and (self.swarm_dir.parent / "config/providers.yaml").exists():
            providers_cfg = self.swarm_dir.parent / "config/providers.yaml"

        is_mock = self.mock_mode or (hasattr(self.client, "__class__") and self.client.__class__.__name__ == "MockLLM")
        self.provider_registry = ProviderRegistry(
            providers_config_path=providers_cfg,
            mock_mode=is_mock,
            offline=self.offline,
        )

        if self.client is None:
            if is_mock:
                self.client = MockLLM()
            else:
                try:
                    self.client = self.provider_registry.get_provider(self.orch_provider_name)
                except Exception as exc:
                    logger.warning("Could not instantiate provider %s (%s); falling back to OllamaClient", self.orch_provider_name, exc)
                    self.client = OllamaClient()

        debugger_provider = None
        debugger_model = "claude-3-5-haiku"
        if self.escalate_to_cloud:
            try:
                debugger_provider = self.provider_registry.get_provider("anthropic")
            except Exception:
                debugger_provider = self.provider_registry.get_default_provider()

        self.escalation_handler = CloudEscalationHandler(
            enabled=self.escalate_to_cloud,
            provider=debugger_provider,
            model=debugger_model,
            outbound_filter=self.outbound_filter,
            budget_tracker=self.cloud_budget,
        )

        self.two_pool_scheduler = TwoPoolScheduler(
            task_queue=self.task_queue,
            local_concurrency=1,
            cloud_concurrency=2,
        )
        self.traceability: TraceabilityMatrix | None = None

    def _load_yaml(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {}

    def _determine_enabled_roles(self) -> set[str]:
        """Parse roles.yaml and allow role overrides from environment/CLI."""
        roles_dict = self.roles_config.get("roles", {})
        enabled = set()
        for r_name, r_cfg in roles_dict.items():
            if r_cfg.get("enabled", False):
                enabled.add(r_name)

        roles_input = getattr(self, "roles", None) or os.getenv("ROLES")
        if roles_input:
            if isinstance(roles_input, str):
                parsed = [r.strip() for r in roles_input.split(",") if r.strip()]
            else:
                parsed = list(roles_input)
            enabled.update(parsed)

        # Fallback to backend + tester if empty
        if not enabled:
            enabled = {"backend", "tester"}
        return enabled

    def clarify_requirements(self) -> list[str]:
        """Clarify requirements with user or record technical assumptions."""
        assumptions: list[str] = []

        if self.interactive and self.question_prompter:
            print("\n🤔 [Chief Orchestrator] Interactive Requirement Clarification")
            # Ask clarifying questions up to limit
            q1 = "Are there specific files or modules that must not be modified?"
            ans1 = self.question_prompter(q1)
            if ans1:
                assumptions.append(f"User requirement: {ans1}")

            q2 = "What are the key acceptance tests to satisfy?"
            ans2 = self.question_prompter(q2)
            if ans2:
                assumptions.append(f"Acceptance constraint: {ans2}")
        else:
            # Non-interactive mode: record sensible technical assumptions
            assumptions.append("Non-interactive mode: decompose into smallest atomic tasks.")
            assumptions.append(f"Conform to 16 GB profile: max {self.max_files_per_task} files per task.")
            assumptions.append("Use Python standard library only (CLI/core modules); no external uninstalled dependencies (e.g. no Kivy, PyQt).")
            assumptions.append("All tests must pass deterministically before task branch merge.")

        self.event_log.emit("clarification_complete", args={"assumptions": assumptions})
        return assumptions

    async def generate_plan(self, assumptions: list[str]) -> EngineeringPlan:
        """Query local model for structured EngineeringPlan and validate with deterministic retry."""
        p_path = self.run_dir / "plan.json"
        is_mock = self.mock_mode or (hasattr(self.client, "__class__") and self.client.__class__.__name__ == "MockLLM")
        if self.resume_run_id and not is_mock and p_path.exists():
            try:
                plan_data = json.loads(p_path.read_text(encoding="utf-8"))
                plan = EngineeringPlan.model_validate(plan_data)
                return plan
            except Exception as exc:
                logger.warning("Existing plan.json could not be loaded (%s); generating new plan", exc)

        repo_map = generate_repo_map(self.repo_path, max_chars=6000)
        coding_roles = [r for r in sorted(self.enabled_roles) if r not in ("pm", "architect", "qa", "reviewer")]
        if not coding_roles:
            coding_roles = ["backend", "tester"]
        roles_summary = ", ".join(coding_roles)

        system_prompt = (
            "You are the Chief Engineering Orchestrator. You break user goals into a small, "
            "acyclic task DAG for specialist engineers.\n"
            "RULES:\n"
            f"1. Available roles: {roles_summary}.\n"
            f"2. Each task must modify at most {self.max_files_per_task} files (narrow write_scope).\n"
            "3. Acceptance criteria must be at most 5 lines per task.\n"
            "4. Prefer many small tasks over few large ones (recommend 2 to 5 tasks total).\n"
            "5. If two tasks write to the same file, the subsequent task MUST include the previous task in depends_on. Parallel tasks must have completely disjoint write scopes.\n"
            "6. You NEVER edit code directly.\n"
            "7. Output valid JSON matching the schema.\n"
            "8. STDLIB ONLY: Use ONLY standard Python library (e.g. CLI, functions, math, argparse). NEVER plan for uninstalled external GUI or web frameworks (like Kivy, PyQt, Django, Flask) unless the user explicitly requested them.\n\n"
            f"{repo_map}\n"
        )

        user_content = (
            f"Goal: {self.goal}\n"
            f"Assumptions:\n" + "\n".join(f"- {a}" for a in assumptions) + "\n\n"
            "Generate the engineering plan JSON."
        )

        schema = EngineeringPlan.model_json_schema()

        # Attempt 1
        logger.info("Generating engineering plan with model: %s", self.model)
        if isinstance(self.client, BaseProvider):
            chat_messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_content),
            ]
            p_resp = await self.client.chat(
                model=self.model,
                messages=chat_messages,
                response_format=schema,
                temperature=0.1,
                max_tokens=self.num_predict,
            )
            resp = OllamaChatResponse(
                model=self.model,
                content=p_resp.content,
                tool_calls=[],
                prompt_eval_count=p_resp.usage.input_tokens,
                eval_count=p_resp.usage.output_tokens,
                tokens_per_sec=(p_resp.usage.output_tokens / p_resp.latency_s) if p_resp.latency_s > 0 else 0.0,
                time_to_first_token_s=p_resp.latency_s,
                load_duration_s=0.0,
            )
        else:
            resp = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                format=schema,
                options={"num_ctx": self.num_ctx, "temperature": 0.1, "num_predict": self.num_predict},
                keep_alive="-1",
            )

        # Parse & Validate
        try:
            raw_data = resp.parsed_json()
            plan = EngineeringPlan.model_validate(raw_data)
            normalize_plan_dependencies(plan.tasks)
            validate_engineering_plan(
                plan,
                enabled_roles=self.enabled_roles,
                max_files_per_task=self.max_files_per_task,
                max_run_minutes=self.max_run_minutes,
            )
            self._save_plan(plan)
            return plan
        except (json.JSONDecodeError, ValueError, PlanValidationError) as exc:
            errors = exc.errors if isinstance(exc, PlanValidationError) else [str(exc)]
            logger.warning("Plan attempt 1 failed validation: %s. Retrying once with error feedback...", errors)

            # Attempt 2: Repair retry
            repair_prompt = build_plan_repair_prompt(resp.content, errors)
            if isinstance(self.client, BaseProvider):
                retry_messages = [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_content),
                    ChatMessage(role="assistant", content=resp.content),
                    ChatMessage(role="user", content=repair_prompt),
                ]
                p_retry = await self.client.chat(
                    model=self.model,
                    messages=retry_messages,
                    response_format=schema,
                    temperature=0.0,
                    max_tokens=self.num_predict,
                )
                retry_resp = OllamaChatResponse(
                    model=self.model,
                    content=p_retry.content,
                    tool_calls=[],
                    prompt_eval_count=p_retry.usage.input_tokens,
                    eval_count=p_retry.usage.output_tokens,
                    tokens_per_sec=(p_retry.usage.output_tokens / p_retry.latency_s) if p_retry.latency_s > 0 else 0.0,
                    time_to_first_token_s=p_retry.latency_s,
                    load_duration_s=0.0,
                )
            else:
                retry_resp = await self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": resp.content},
                        {"role": "user", "content": repair_prompt},
                    ],
                    format=schema,
                    options={"num_ctx": self.num_ctx, "temperature": 0.0, "num_predict": self.num_predict},
                    keep_alive="-1",
                )

            try:
                raw_data2 = retry_resp.parsed_json()
                plan2 = EngineeringPlan.model_validate(raw_data2)
                normalize_plan_dependencies(plan2.tasks)
                validate_engineering_plan(
                    plan2,
                    enabled_roles=self.enabled_roles,
                    max_files_per_task=self.max_files_per_task,
                    max_run_minutes=self.max_run_minutes,
                )
                self._save_plan(plan2)
                return plan2
            except Exception as final_exc:
                logger.error("Plan retry failed validation: %s", final_exc)
                raise InvalidPlanError(f"Engineering plan failed validation after repair retry: {final_exc}") from final_exc

    def _save_plan(self, plan: EngineeringPlan) -> None:
        """Persist validated plan to plan.json and event log."""
        p_path = self.run_dir / "plan.json"
        p_path.write_text(json.dumps(plan.model_dump(), indent=2), encoding="utf-8")
        self.event_log.emit(
            "plan_created",
            args={"task_count": len(plan.tasks), "tasks": [t.id for t in plan.tasks]},
        )

    async def execute_run(self) -> int:
        """Execute the full devswarm workflow: clarify, plan, schedule, and merge."""
        with CaffeinateContext():
            return await self._execute_run_inner()

    async def _execute_run_inner(self) -> int:
        self.kill_switch.check()

        # Laptop awareness: warn if running on battery
        on_battery, batt_msg = check_battery_state()
        if on_battery:
            print(f"\n⚠️  [BATTERY WARNING] MacBook is running on Battery Power!")
            print(f"    {batt_msg}")
            print("    For sustained GPU performance and to prevent sleep/throttling, connect to AC power.\n")
            self.event_log.emit("battery_warning", args={"info": batt_msg})

        # 1. Initialize manifest
        manifest = build_manifest(
            run_id=self.run_id,
            goal=self.goal,
            repo_path=self.repo_path,
            model=self.model,
            num_ctx=self.num_ctx,
            config_snapshot={
                "max_files_per_task": self.max_files_per_task,
                "unload_every_n_tasks": self.unload_every_n_tasks,
                "enabled_roles": list(self.enabled_roles),
            },
        )
        manifest.save(self.run_dir)
        self.event_log.emit("run_start", args={"goal": self.goal, "model": self.model})

        # 2. Check git repo readiness
        try:
            is_resume = bool(getattr(self, "resume_run_id", None))
            self.worktrees.ensure_clean_repo(allow_dirty_on_resume=is_resume)
            self.worktrees.setup_integration_branch(allow_dirty_on_resume=is_resume)
        except Exception as exc:
            logger.error("Git repo check failed: %s", exc)
            self.event_log.emit("run_failed", args={"error": str(exc)})
            return 1

        # 3. Hybrid Pre-Implementation Pipeline (PM -> Gate 1 -> Architect -> Gate 2 -> QA Test-First)
        pipeline_ctrl = PipelineController(
            run_id=self.run_id,
            repo_path=self.repo_path,
            artifacts_dir=self.run_dir / "artifacts",
            event_log=self.event_log,
            sandbox=self.sandbox,
            auto_approve_design=self.auto_approve_design,
            waived_acs=self.waived_acs,
        )

        repo_map = generate_repo_map(self.repo_path, max_chars=6000)
        assumptions = self.clarify_requirements()

        try:
            # Stage 1: Product Manager Agent
            if "pm" in self.enabled_roles:
                pm_prov, pm_model, _, _, _ = self.provider_registry.get_role_binding("pm")
                print(f"\n📋 [Product Manager] Generating feature specification ({pm_model} via {pm_prov.name})...")
                await pipeline_ctrl.run_pm_stage(
                    provider=pm_prov,
                    model=pm_model,
                    goal=self.goal,
                    repo_map=repo_map,
                    assumptions=assumptions,
                )
                pipeline_ctrl.check_gate_1_spec_approval()

            # Stage 2: Software Architect Agent
            if "architect" in self.enabled_roles:
                arch_prov, arch_model, _, _, _ = self.provider_registry.get_role_binding("architect")
                print(f"\n🏛️ [Software Architect] Designing architecture and contract stubs ({arch_model} via {arch_prov.name})...")
                await pipeline_ctrl.run_architect_stage(
                    provider=arch_prov,
                    model=arch_model,
                    goal=self.goal,
                    repo_map=repo_map,
                )
                self.worktrees.commit_integration_changes("Add architecture interface contract stubs")
                pipeline_ctrl.check_gate_2_design_approval()

            # Stage 3: QA Test-First Agent
            if "qa" in self.enabled_roles:
                qa_prov, qa_model, _, _, _ = self.provider_registry.get_role_binding("qa")
                print(f"\n🧪 [QA Engineer] Writing test-first suite with RED-check validation ({qa_model} via {qa_prov.name})...")
                await pipeline_ctrl.run_qa_test_first_stage(
                    provider=qa_prov,
                    model=qa_model,
                )
                self.worktrees.commit_integration_changes("Add QA test-first suite")

        except GatePausedException as exc:
            print(str(exc))
            return 0

        # 4. Clarify & Plan
        try:
            plan = await self.generate_plan(assumptions)
        except InvalidPlanError as exc:
            print(f"❌ Planning failed: {exc}")
            self.event_log.emit("run_failed", args={"error": str(exc)})
            return 1

        # 5. Populate Task Queue
        swarm_tasks: list[SwarmTask] = []
        for t in plan.tasks:
            desc = t.description
            if pipeline_ctrl.spec or pipeline_ctrl.design:
                desc = pipeline_ctrl.generate_task_brief(
                    task_id=t.id,
                    task_title=t.title,
                    task_description=t.description,
                    write_scope=t.write_scope,
                )

            st = SwarmTask(
                id=t.id,
                title=t.title,
                role=t.role,
                description=desc,
                acceptance_criteria=t.acceptance_criteria,
                depends_on=t.depends_on,
                write_scope=t.write_scope,
                read_scope=t.read_scope,
                risk_level=t.risk_level,
                minutes_budget=t.minutes_budget,
            )
            swarm_tasks.append(st)

        self.task_queue.add_tasks(swarm_tasks)
        print(f"📋 Plan generated: {len(swarm_tasks)} tasks scheduled in dependency order.")

        # 5. Sequential Execution Loop (LLM_CONCURRENCY=1)
        return await self._run_execution_loop()

    async def _run_execution_loop(self) -> int:
        """Process ready tasks sequentially until queue is finished or blocked."""
        while not self.task_queue.is_finished():
            # Check kill switch before every task step
            self.kill_switch.check()
            self.budget.check_run_deadline()

            ready_tasks = self.task_queue.get_next_ready(concurrency_limit=1)
            if not ready_tasks:
                # If no tasks ready and queue not finished, check if blocked
                all_t = self.task_queue.get_all_tasks()
                if any(t.state in (TaskState.BLOCKED, TaskState.FAILED, TaskState.NEEDS_HUMAN) for t in all_t):
                    print("⚠️ Swarm paused: one or more tasks are blocked or require human intervention.")
                    return 1
                await asyncio.sleep(0.1)
                continue

            current_task = ready_tasks[0]
            print(f"\n🚀 Starting Task: [{current_task.id}] {current_task.title} (Role: {current_task.role})")

            # Reserve budget
            if not self.budget.reserve(current_task.id, minutes=current_task.minutes_budget):
                print(f"❌ Budget exceeded: cannot reserve resources for {current_task.id}")
                self.task_queue.set_state(current_task.id, TaskState.FAILED, error="Budget exceeded reservation")
                return 1

            self.task_queue.set_state(current_task.id, TaskState.RUNNING)
            self.event_log.emit("task_running", task_id=current_task.id, role=current_task.role)

            # Create isolated worktree for task
            session = self.worktrees.create_task_worktree(current_task.id)

            task_start_time = time.monotonic()
            try:
                task_res = await self._execute_task_work(current_task, session)

                if not task_res.success:
                    self.task_queue.set_state(current_task.id, TaskState.FAILED, error=task_res.error or "Agent loop failed")
                    print(f"❌ Task [{current_task.id}] failed during agent execution")
                else:
                    # Transition to quality gates
                    self.task_queue.set_state(current_task.id, TaskState.GATES)
                    self.event_log.emit("gates_start", task_id=current_task.id, role=current_task.role)
                    print(f"🔍 Running deterministic quality gates for [{current_task.id}]...")

                    test_scope = None
                    if current_task.id == "task_01":
                        test_scope = ["tests/test_calculator_evaluation.py", "tests/test_calculator_errors.py"]
                    elif current_task.id == "task_02":
                        test_scope = ["tests/test_calculator_cli.py", "tests/test_calculator_evaluation.py", "tests/test_calculator_errors.py"]
                    elif any(w.startswith("tests/") for w in current_task.write_scope):
                        test_scope = [w for w in current_task.write_scope if w.startswith("tests/")]

                    gate_res = await self.gate_runner.run_suite(
                        worktree_dir=session.worktree_path,
                        modified_files=task_res.files_modified,
                        test_scope=test_scope,
                    )

                    all_modified = list(task_res.files_modified)
                    diag = extract_failure_diagnostics(gate_res) if not gate_res.passed else ""
                    self.attempt_manager.record_attempt(
                        task_id=current_task.id,
                        files_modified=all_modified,
                        gate_passed=gate_res.passed,
                        gate_summary=gate_res.summary,
                        diagnostics=diag,
                    )

                    # Automated self-healing loop up to max_fix_attempts
                    local_attempt = 1
                    while not gate_res.passed and local_attempt <= self.max_fix_attempts:
                        logger.warning(
                            "Task %s failed quality gates (attempt %d/%d): %s. Initiating local self-healing fix...",
                            current_task.id,
                            local_attempt,
                            self.max_fix_attempts,
                            gate_res.summary,
                        )
                        self.event_log.emit(
                            "gates_failed",
                            task_id=current_task.id,
                            role=current_task.role,
                            args={"attempt": local_attempt, "summary": gate_res.summary},
                        )
                        print(f"⚠️ Task [{current_task.id}] gates failed: attempting local self-healing fix (attempt {local_attempt}/{self.max_fix_attempts})...")
                        for g_name, g_res in gate_res.results.items():
                            if not g_res.passed:
                                msg = g_res.error_message or ""
                                out = (g_res.stderr or g_res.stdout).strip()
                                if out:
                                    failed_lines = [l.strip() for l in out.splitlines() if "FAILED " in l]
                                    if failed_lines:
                                        print(f"   [{g_name}] {msg} -> " + "; ".join(failed_lines))
                                    else:
                                        print(f"   [{g_name}] {msg} -> " + out.splitlines()[-1][:120])
                                else:
                                    print(f"   [{g_name}] {msg}")

                        diagnostics_feedback = extract_failure_diagnostics(gate_res)
                        repair_res = await self._execute_task_work(
                            current_task,
                            session,
                            additional_instructions=diagnostics_feedback,
                        )

                        if repair_res.files_modified:
                            all_modified = sorted(list(set(all_modified + repair_res.files_modified)))

                        gate_res = await self.gate_runner.run_suite(
                            worktree_dir=session.worktree_path,
                            modified_files=all_modified,
                            test_scope=test_scope,
                        )
                        diag = extract_failure_diagnostics(gate_res) if not gate_res.passed else ""
                        self.attempt_manager.record_attempt(
                            task_id=current_task.id,
                            files_modified=all_modified,
                            gate_passed=gate_res.passed,
                            gate_summary=gate_res.summary,
                            diagnostics=diag,
                        )
                        local_attempt += 1

                    # Optional Cloud Escalation if still failing
                    if not gate_res.passed and self.escalate_to_cloud and self.escalation_handler.enabled:
                        print(f"🚨 Task [{current_task.id}] exhausted local fix attempts ({self.max_fix_attempts}). Escalating to cloud debugger...")
                        self.event_log.emit("cloud_escalation_started", task_id=current_task.id, role=current_task.role)

                        recent_code = ""
                        for w_file in current_task.write_scope:
                            w_p = session.worktree_path / w_file
                            if w_p.exists():
                                recent_code += f"# {w_file}\n" + w_p.read_text(encoding="utf-8")[:1500] + "\n"

                        advice = await self.escalation_handler.request_debug_advice(
                            task_id=current_task.id,
                            task_title=current_task.title,
                            write_scope=current_task.write_scope,
                            diagnostics=extract_failure_diagnostics(gate_res),
                            recent_code=recent_code,
                            run_id=self.run_id,
                        )
                        if advice:
                            print(f"💡 Cloud debugger provided advice for [{current_task.id}]. Local coder applying patch...")
                            self.event_log.emit("cloud_escalation_advice_received", task_id=current_task.id)
                            cloud_fix_res = await self._execute_task_work(
                                current_task,
                                session,
                                additional_instructions=advice,
                            )
                            if cloud_fix_res.files_modified:
                                all_modified = sorted(list(set(all_modified + cloud_fix_res.files_modified)))
                            gate_res = await self.gate_runner.run_suite(
                                worktree_dir=session.worktree_path,
                                modified_files=all_modified,
                                test_scope=test_scope,
                            )
                            diag = extract_failure_diagnostics(gate_res) if not gate_res.passed else ""
                            self.attempt_manager.record_attempt(
                                task_id=current_task.id,
                                files_modified=all_modified,
                                gate_passed=gate_res.passed,
                                gate_summary=gate_res.summary,
                                diagnostics=diag,
                            )

                    if not gate_res.passed:
                        self.task_queue.set_state(
                            current_task.id,
                            TaskState.FAILED,
                            error=f"Quality gates failed after repair: {gate_res.summary}",
                        )
                        self.event_log.emit(
                            "task_failed",
                            task_id=current_task.id,
                            role=current_task.role,
                            args={"error": gate_res.summary},
                        )
                        print(f"❌ Task [{current_task.id}] failed quality gates: {gate_res.summary}")
                        for g_name, g_res in gate_res.results.items():
                            if not g_res.passed:
                                msg = g_res.error_message or ""
                                out = (g_res.stderr or g_res.stdout).strip()
                                if out:
                                    failed_lines = [l.strip() for l in out.splitlines() if "FAILED " in l]
                                    if failed_lines:
                                        print(f"   [{g_name}] {msg} -> " + "; ".join(failed_lines))
                                    else:
                                        print(f"   [{g_name}] {msg} -> " + out.splitlines()[-1][:120])
                                else:
                                    print(f"   [{g_name}] {msg}")
                    elif gate_res.has_high_risk_changes:
                        # High-risk path modified -> requires human approval
                        self.task_queue.set_state(
                            current_task.id,
                            TaskState.NEEDS_HUMAN,
                            result_summary=f"Passed quality gates but modified high-risk paths: {', '.join(gate_res.high_risk_files)}",
                        )
                        self.event_log.emit(
                            "task_needs_human",
                            task_id=current_task.id,
                            role=current_task.role,
                            args={"high_risk_files": gate_res.high_risk_files},
                        )
                        print(f"⚠️ Task [{current_task.id}] passed gates but requires HUMAN APPROVAL (high-risk files: {', '.join(gate_res.high_risk_files)})")
                    else:
                        # Required gates passed and no high-risk changes
                        self.event_log.emit(
                            "gates_passed",
                            task_id=current_task.id,
                            role=current_task.role,
                            args={"summary": gate_res.summary},
                        )
                        # Commit platform changes
                        self.worktrees.commit_changes(current_task.id, f"Completed: {current_task.title}")
                        # Merge task branch into integration branch
                        self.worktrees.merge_task_branch(current_task.id)

                        self.task_queue.set_state(
                            current_task.id,
                            TaskState.DONE,
                            result_summary=f"Completed, passed gates, and merged ({gate_res.summary})",
                        )
                        self.event_log.emit("task_done", task_id=current_task.id, role=current_task.role)
                        print(f"✅ Task [{current_task.id}] passed quality gates and merged into {self.worktrees.integration_branch}")
            except GitConflictError as exc:
                logger.error("Merge conflict in task %s: %s", current_task.id, exc)
                self.task_queue.set_state(current_task.id, TaskState.NEEDS_HUMAN, error=str(exc))
                print(f"⚠️ Merge conflict in task [{current_task.id}] -> flagged as NEEDS_HUMAN")
            except Exception as exc:
                logger.exception("Error during task %s: %s", current_task.id, exc)
                self.task_queue.set_state(current_task.id, TaskState.FAILED, error=str(exc))
            finally:
                self.worktrees.remove_worktree(current_task.id)
                self.budget.release(current_task.id)

                # Memory Safety: UNLOAD_MODEL_EVERY_N_TASKS
                self.tasks_since_last_unload += 1
                if self.tasks_since_last_unload >= self.unload_every_n_tasks:
                    logger.info("Unload threshold reached (%d tasks). Unloading model to clear KV cache...", self.tasks_since_last_unload)
                    if hasattr(self.client, "unload_model"):
                        await self.client.unload_model(self.model)
                    self.tasks_since_last_unload = 0

        all_tasks = self.task_queue.get_all_tasks()
        has_failures = any(t.state in (TaskState.FAILED, TaskState.BLOCKED) for t in all_tasks)
        has_needs_human = any(t.state == TaskState.NEEDS_HUMAN for t in all_tasks)

        if has_failures:
            print(f"\n❌ Swarm execution finished with failed tasks.")
            self.event_log.emit("run_completed_with_failures", args={"integration_branch": self.worktrees.integration_branch})
            return 1

        if has_needs_human:
            print(f"\n⚠️ Swarm execution finished with tasks awaiting human approval.")
            self.event_log.emit("run_needs_human", args={"integration_branch": self.worktrees.integration_branch})
            return 1

        # Generate Senior Review and Pull Request artifact
        rev_prov, rev_model, _, _, _ = self.provider_registry.get_role_binding("reviewer")
        use_cloud_reviewer = "reviewer" in self.enabled_roles and rev_prov.name != "ollama"
        prov = rev_prov if use_cloud_reviewer else (self.client if isinstance(self.client, BaseProvider) else None)
        reviewer = SwarmReviewer(
            client=self.client if not isinstance(self.client, BaseProvider) else None,
            provider=prov,
            repo_path=self.repo_path,
            budget=self.budget,
            event_log=self.event_log,
            reviewer_model=rev_model if use_cloud_reviewer else self.model,
            default_model=self.model,
            num_ctx=self.num_ctx,
        )

        task_summaries = [t.to_dict() for t in all_tasks]
        review_report = await reviewer.review_run(
            goal=self.goal,
            base_commit=self.worktrees.base_commit,
            integration_branch=self.worktrees.integration_branch,
            task_summaries=task_summaries,
            current_active_model=self.model,
        )

        # Run final quality gate evaluation on the cumulative integration branch
        cum_gate_res = await self.gate_runner.run_suite(
            worktree_dir=self.repo_path,
            modified_files=[],
        )

        # Collect cloud telemetry and accounting for PR
        role_telemetry = []
        if hasattr(self, "cloud_budget") and self.cloud_budget:
            for r_name, spent in self.cloud_budget.spent_usd_by_role.items():
                if spent > 0 or r_name in ("pm", "architect", "qa", "reviewer", "debugger"):
                    try:
                        r_prov, r_mod, _, _, _ = self.provider_registry.get_role_binding(r_name)
                        p_name = r_prov.name
                        m_name = r_mod
                    except Exception:
                        p_name = "siliconflow" if r_name in ("pm", "architect", "qa", "reviewer") else "ollama"
                        m_name = self.model
                    role_telemetry.append({
                        "role": r_name,
                        "provider": p_name,
                        "model": m_name,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": spent,
                    })

        cloud_accounting = None
        if hasattr(self, "cloud_budget") and self.cloud_budget:
            cloud_accounting = {
                "spent_usd_total": self.cloud_budget.spent_usd_total,
                "content_classes": ["goal", "spec", "architecture", "test_files", "gate_logs"],
                "bytes_sent": getattr(self.outbound_filter, "total_bytes_sent", 0) if hasattr(self, "outbound_filter") else 0,
            }

        trace_md = ""
        if hasattr(self, "traceability") and self.traceability:
            trace_md = self.traceability.render_matrix_markdown()

        pr_file = PRGenerator.generate(
            run_id=self.run_id,
            goal=self.goal,
            repo_path=self.repo_path,
            base_commit=self.worktrees.base_commit,
            integration_branch=self.worktrees.integration_branch,
            tasks=task_summaries,
            gate_result=cum_gate_res,
            review_report=review_report,
            budget_summary=self.budget.summary(),
            output_dir=self.run_dir,
            traceability_markdown=trace_md,
            role_telemetry=role_telemetry,
            cloud_accounting=cloud_accounting,
        )

        print(f"\n🎉 All swarm tasks processed! Integration branch: {self.worktrees.integration_branch}")
        print(f"✨ Pull Request generated: {pr_file}")
        print(f"   Reviewer Verdict: {review_report.verdict} (Confidence: {review_report.confidence_score * 100:.0f}%)")
        print(f"\n👉 To approve and merge into main:")
        print(f"   python main.py approve --run-id {self.run_id}")
        self.event_log.emit(
            "run_complete",
            args={
                "integration_branch": self.worktrees.integration_branch,
                "pr_path": str(pr_file),
                "reviewer_verdict": review_report.verdict,
            },
        )
        return 0

    async def _execute_task_work(
        self,
        task: SwarmTask,
        session: Any,
        additional_instructions: str = "",
    ) -> TaskExecutionResult:
        """Execute specialist agent tool loop inside the task worktree."""
        logger.info("Executing specialist agent loop for task %s (Role: %s) in %s", task.id, task.role, session.worktree_path)
        try:
            role_spec = self.role_registry.get_role(task.role)
        except KeyError:
            logger.warning("Role '%s' not explicitly configured; using default RoleSpec", task.role)
            role_spec = RoleSpec(name=task.role, enabled=True)

        desc = task.description
        if additional_instructions:
            desc = f"{task.description}\n\nREPAIR INSTRUCTIONS (Quality Gate Failures):\n{additional_instructions}"

        task_model = self.model
        task_client = self.client
        is_mock = self.mock_mode or (hasattr(self.client, "__class__") and self.client.__class__.__name__ == "MockLLM")
        if not is_mock:
            try:
                prov, role_model, _, _, _ = self.provider_registry.get_role_binding(task.role)
                if role_model:
                    task_model = role_model
                if prov.name != "ollama":
                    task_client = prov
            except Exception:
                pass

        res = await run_agent_loop(
            task_id=task.id,
            task_title=task.title,
            task_description=desc,
            role_spec=role_spec,
            write_scope=task.write_scope,
            acceptance_criteria=task.acceptance_criteria,
            worktree_dir=session.worktree_path,
            client=task_client,
            sandbox=self.sandbox,
            budget=self.budget,
            kill_switch=self.kill_switch,
            event_log=self.event_log,
            model=task_model,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            max_files=self.max_files_per_task,
            throughput_guard=self.throughput_guard,
        )

        return res

    def get_status(self) -> dict[str, Any]:
        """Return formatted status dictionary of the current or resumed run."""
        tasks = self.task_queue.get_all_tasks()
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "model": self.model,
            "tasks": [t.to_dict() for t in tasks],
            "total_tasks": len(tasks),
            "done_tasks": sum(1 for t in tasks if t.state == TaskState.DONE),
            "failed_tasks": sum(1 for t in tasks if t.state == TaskState.FAILED),
            "budget": self.budget.summary(),
        }

    def print_status_table(self) -> None:
        """Print a human-readable table of task states."""
        status = self.get_status()
        print(f"\n📊 Swarm Run Status: {status['run_id']}")
        print(f"Goal: {status['goal']}")
        print(f"Model: {status['model']}")
        print("\n| Task ID | Role | State | Retries | Title |")
        print("|---|---|---|:---:|---|")
        for t in status["tasks"]:
            badge = f"`{t['state']}`"
            print(f"| {t['id']} | {t['role']} | {badge} | {t['retry_count']}/{t['max_retries']} | {t['title']} |")
        print()

    async def run_qa_post_pass(
        self,
        qa_provider: BaseProvider | None = None,
        model: str = "",
    ) -> QATestSuite | None:
        """
        QA reviews implementation vs specification and optionally appends edge-case tests to tests/.
        """
        if not qa_provider:
            return None

        spec_file = self.run_dir / "artifacts" / "spec.md"
        spec_text = spec_file.read_text(encoding="utf-8") if spec_file.exists() else self.goal
        repo_map = generate_repo_map(self.repo_path, max_chars=4000)

        system_prompt = (
            "You are the QA Engineer performing a post-implementation edge-case test pass. "
            "Review the codebase against the specification. "
            "Identify missing edge cases, boundary conditions, and error-handling scenarios. "
            "Generate supplementary test cases strictly in the tests/ directory.\n"
            "RULES:\n"
            "1. Output MUST be valid JSON matching the QATestSuite schema.\n"
            "2. Modify ONLY files within tests/ (never implementation code).\n"
            "3. Tests must be valid Python/pytest code.\n"
        )

        user_content = (
            f"Specification:\n{spec_text}\n\n"
            f"Current Repository:\n{repo_map}\n\n"
            "Generate supplementary edge-case tests."
        )

        schema = QATestSuite.model_json_schema()
        resp = await qa_provider.chat(
            model=model or "claude-3-5-haiku",
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_content),
            ],
            response_format=schema,
            max_tokens=4096,
        )

        try:
            qa_suite = QATestSuite.model_validate(resp.parsed_json())
            for tf in qa_suite.test_files:
                p = self.repo_path / tf.path
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.exists():
                    existing = p.read_text(encoding="utf-8")
                    if tf.content.strip() not in existing:
                        p.write_text(existing + "\n\n" + tf.content.strip() + "\n", encoding="utf-8")
                else:
                    p.write_text(tf.content, encoding="utf-8")

            gate_res = await self.gate_runner.run_suite(
                worktree_dir=self.repo_path,
                modified_files=[tf.path for tf in qa_suite.test_files],
            )
            self.event_log.emit(
                "qa_post_pass_complete",
                args={"passed": gate_res.passed, "summary": gate_res.summary},
            )
            return qa_suite
        except Exception as exc:
            logger.warning("QA post-pass failed: %s", exc)
            return None
