"""
tests/devswarm/test_taskqueue.py — Unit tests for SQLite task queue, DAG transitions, and crash recovery.
"""

from devswarm.platform.taskqueue import TaskQueue, SwarmTask, TaskState


def test_task_queue_dag_resolution(tmp_path):
    db_file = tmp_path / "tasks.db"
    queue = TaskQueue(db_file)

    task1 = SwarmTask(id="task-1", title="Task 1", role="backend", description="Base task")
    task2 = SwarmTask(id="task-2", title="Task 2", role="tester", description="Dependent task", depends_on=["task-1"])

    queue.add_tasks([task1, task2])

    # task1 has no deps -> READY, task2 has dep -> PENDING
    assert queue.get_task("task-1").state == TaskState.READY
    assert queue.get_task("task-2").state == TaskState.PENDING

    # Transition task1 to RUNNING, then DONE
    queue.set_state("task-1", TaskState.RUNNING)
    assert queue.get_task("task-2").state == TaskState.PENDING

    queue.set_state("task-1", TaskState.DONE)
    # task2 should now be automatically resolved to READY
    assert queue.get_task("task-2").state == TaskState.READY


def test_task_queue_failure_propagation(tmp_path):
    db_file = tmp_path / "tasks.db"
    queue = TaskQueue(db_file)

    task1 = SwarmTask(id="task-1", title="Task 1", role="backend", description="Base task")
    task2 = SwarmTask(id="task-2", title="Task 2", role="tester", description="Dependent task", depends_on=["task-1"])

    queue.add_tasks([task1, task2])
    assert queue.get_task("task-2").state == TaskState.PENDING

    # Fail task1
    queue.set_state("task-1", TaskState.FAILED, error="Syntax error in base module")

    # task2 should now transition to BLOCKED
    assert queue.get_task("task-2").state == TaskState.BLOCKED
    assert "Blocked by upstream" in queue.get_task("task-2").error


def test_task_queue_concurrency_limit(tmp_path):
    db_file = tmp_path / "tasks.db"
    queue = TaskQueue(db_file)

    t1 = SwarmTask(id="t1", title="T1", role="backend", description="task 1")
    t2 = SwarmTask(id="t2", title="T2", role="backend", description="task 2")
    t3 = SwarmTask(id="t3", title="T3", role="backend", description="task 3")

    queue.add_tasks([t1, t2, t3])

    # Concurrency limit = 1: should return only 1 task
    ready = queue.get_next_ready(concurrency_limit=1)
    assert len(ready) == 1
    assert ready[0].id == "t1"

    # Mark t1 as RUNNING
    queue.set_state("t1", TaskState.RUNNING)

    # Next ready with concurrency limit = 1: slot is occupied, should return empty
    assert len(queue.get_next_ready(concurrency_limit=1)) == 0

    # Next ready with concurrency limit = 2: 1 slot left, should return t2
    ready2 = queue.get_next_ready(concurrency_limit=2)
    assert len(ready2) == 1
    assert ready2[0].id == "t2"


def test_task_queue_crash_recovery(tmp_path):
    db_file = tmp_path / "tasks.db"
    queue = TaskQueue(db_file)

    t1 = SwarmTask(id="t1", title="T1", role="backend", description="task 1", max_retries=2)
    t2 = SwarmTask(id="t2", title="T2", role="backend", description="task 2", max_retries=1, retry_count=1)

    queue.add_tasks([t1, t2])
    queue.set_state("t1", TaskState.RUNNING)
    queue.set_state("t2", TaskState.GATES)

    # Simulate crash and recover
    recovered = queue.recover_after_crash()
    assert recovered == 2

    # t1 had retry_count=0 -> now 1, state reset to READY
    t1_after = queue.get_task("t1")
    assert t1_after.state == TaskState.READY
    assert t1_after.retry_count == 1
    assert "Crash recovery" in t1_after.error

    # t2 had retry_count=1, max_retries=1 -> now 2, exceeds max_retries -> FAILED
    t2_after = queue.get_task("t2")
    assert t2_after.state == TaskState.FAILED
    assert t2_after.retry_count == 2

