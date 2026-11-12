"""
devswarm.scheduler — Two-pool hybrid resource scheduler.

Manages:
  1. LocalPool: Concurrency 1 (Ollama Metal GPU; local coding agents).
  2. CloudPool: Concurrency N (configurable; rate-limit aware; cloud specialist roles).
Allows independent cloud steps (e.g. QA writing test suites) to run concurrently
with local coder execution without overloading the local machine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from devswarm.platform.taskqueue import SwarmTask, TaskQueue, TaskState

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTaskResult:
    task_id: str
    role: str
    pool: str  # "local" or "cloud"
    success: bool
    output: Any = None
    error: str = ""


class TwoPoolScheduler:
    """Schedules tasks across separated Local (concurrency 1) and Cloud (concurrency N) pools."""

    def __init__(
        self,
        task_queue: TaskQueue,
        local_concurrency: int = 1,
        cloud_concurrency: int = 2,
    ) -> None:
        self.task_queue = task_queue
        self.local_concurrency = local_concurrency
        self.cloud_concurrency = cloud_concurrency

        self.local_semaphore = asyncio.Semaphore(local_concurrency)
        self.cloud_semaphore = asyncio.Semaphore(cloud_concurrency)

        self.running_tasks: set[str] = set()

    def determine_pool(self, role: str, provider: str = "ollama") -> str:
        """Categorize task into local or cloud execution pool."""
        if provider.lower() == "ollama" or role in ("backend", "tester", "coder"):
            return "local"
        return "cloud"

    async def execute_task_with_pool(
        self,
        task: SwarmTask,
        provider: str,
        executor_func: Callable[[SwarmTask], Coroutine[Any, Any, Any]],
    ) -> ScheduledTaskResult:
        """Run task inside its appropriate concurrency semaphore."""
        pool = self.determine_pool(task.role, provider)
        sem = self.local_semaphore if pool == "local" else self.cloud_semaphore

        async with sem:
            self.running_tasks.add(task.id)
            self.task_queue.set_state(task.id, TaskState.RUNNING)
            logger.info("Dispatched [%s] (%s) to %s pool (Active running: %d)", task.id, task.role, pool.upper(), len(self.running_tasks))

            try:
                result = await executor_func(task)
                success = getattr(result, "success", True)
                err = getattr(result, "error", "")
                st = TaskState.GATES if success else TaskState.FAILED
                self.task_queue.set_state(task.id, st, error=err)
                return ScheduledTaskResult(
                    task_id=task.id,
                    role=task.role,
                    pool=pool,
                    success=success,
                    output=result,
                    error=err,
                )
            except Exception as exc:
                logger.error("Task [%s] crashed in %s pool: %s", task.id, pool, exc)
                self.task_queue.set_state(task.id, TaskState.FAILED, error=str(exc))
                return ScheduledTaskResult(
                    task_id=task.id,
                    role=task.role,
                    pool=pool,
                    success=False,
                    error=str(exc),
                )
            finally:
                self.running_tasks.discard(task.id)

    def get_ready_tasks_by_pool(self, role_providers: dict[str, str]) -> tuple[list[SwarmTask], list[SwarmTask]]:
        """Separate ready tasks into local candidates and cloud candidates."""
        ready = [t for t in self.task_queue.get_all_tasks() if t.state == TaskState.READY]
        local_tasks: list[SwarmTask] = []
        cloud_tasks: list[SwarmTask] = []

        for t in ready:
            prov = role_providers.get(t.role, "ollama")
            if self.determine_pool(t.role, prov) == "local":
                local_tasks.append(t)
            else:
                cloud_tasks.append(t)

        return local_tasks, cloud_tasks
