"""devswarm.agents — Specialist roles, planning schemas, and execution loops."""

from devswarm.agents.plan_schema import (
    TaskPlanItem,
    EngineeringPlan,
    PlanValidationError,
    validate_engineering_plan,
)
from devswarm.agents.roles import RoleSpec, RoleRegistry
from devswarm.agents.prompts import build_system_prompt
from devswarm.agents.context import AgentContext
from devswarm.agents.parser import extract_tool_calls, ParsedToolCall
from devswarm.agents.loop import run_agent_loop, TaskExecutionResult

__all__ = [
    "TaskPlanItem",
    "EngineeringPlan",
    "PlanValidationError",
    "validate_engineering_plan",
    "RoleSpec",
    "RoleRegistry",
    "build_system_prompt",
    "AgentContext",
    "extract_tool_calls",
    "ParsedToolCall",
    "run_agent_loop",
    "TaskExecutionResult",
]
