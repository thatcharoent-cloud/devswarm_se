"""
devswarm.platform.guards — Safety guards, loop detectors, kill switch, and resource monitoring.

Features:
  - Kill Switch: checks `.swarm/STOP` before every agent action; raises KillSwitchTriggered.
  - Loop Detector: aborts task if the exact same tool call and arguments are repeated 3 times.
  - Throughput Collapse Guard: pauses run if tokens/sec < MIN_TOKENS_PER_SEC for N consecutive calls.
  - Memory Pressure Guard: checks system swap / memory pressure.
  - Caffeinate Wrapper: wraps runs with macOS `caffeinate -i` so the Mac does not sleep.
  - Battery Monitor: warns if laptop is running on battery power.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from devswarm.platform.budget import SwarmBudget, BudgetExceededError

logger = logging.getLogger(__name__)

DEFAULT_KILL_SWITCH_FILE = ".swarm/STOP"
DEFAULT_LOOP_LIMIT = 3
DEFAULT_MIN_TOKENS_PER_SEC = 5.0
DEFAULT_COLLAPSE_STREAK = 3


class GuardError(Exception):
    """Base exception for guard checks."""


class KillSwitchTriggered(GuardError):
    """Raised when the .swarm/STOP file is detected."""


KillSwitchTriggeredError = KillSwitchTriggered


class LoopDetectedError(GuardError):
    """Raised when an agent repeats the same tool call with identical arguments 3 times."""


class ThroughputCollapsedError(GuardError):
    """Raised when local inference throughput falls below acceptable threshold for consecutive calls."""


class CriticalMemoryPressureError(GuardError):
    """Raised when system memory pressure is critical on macOS."""


class LoopDetector:
    """Tracks tool invocation signatures within a task to prevent infinite loops."""

    def __init__(self, limit: int = DEFAULT_LOOP_LIMIT, threshold: int | None = None) -> None:
        self.limit = threshold if threshold is not None else limit
        self.history: list[tuple[str, str]] = []  # [(tool_name, serialized_args)]

    def record(self, tool_name: str, args: dict[str, Any] | str) -> None:
        norm_args = json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)
        self.history.append((tool_name, norm_args))

    def is_looping(self) -> bool:
        if not self.history:
            return False
        sig = self.history[-1]
        consecutive = 0
        for item in reversed(self.history):
            if item == sig:
                consecutive += 1
            else:
                break
        return consecutive >= self.limit

    def record_and_check(self, tool_name: str, args: dict[str, Any] | str) -> None:
        """Record a tool invocation and raise LoopDetectedError if repeated >= limit."""
        self.record(tool_name, args)
        if self.is_looping():
            msg = (
                f"Loop detected: agent repeated tool '{tool_name}' with identical arguments "
                f"{self.limit} times consecutively. Aborting task."
            )
            logger.error(msg)
            raise LoopDetectedError(msg)

    def reset(self) -> None:
        self.history.clear()


class KillSwitch:
    """Monitors for manual abort signal (.swarm/STOP)."""

    def __init__(self, stop_file: Path | str = DEFAULT_KILL_SWITCH_FILE) -> None:
        self.stop_file = Path(stop_file)

    def check(self) -> None:
        """Check if stop file exists. If so, raise KillSwitchTriggered."""
        if self.stop_file.exists():
            msg = f"Kill switch triggered: '{self.stop_file}' was found. Stopping run cleanly."
            logger.warning(msg)
            raise KillSwitchTriggered(msg)

    def trigger(self, reason: str = "Manual stop requested") -> None:
        """Create the stop file."""
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text(reason, encoding="utf-8")

    def clear(self) -> None:
        """Remove the stop file."""
        if self.stop_file.exists():
            self.stop_file.unlink()

    def remove(self) -> None:
        """Alias for clear()."""
        self.clear()

    def is_triggered(self) -> bool:
        """Return True if stop file exists."""
        return self.stop_file.exists()


class ThroughputGuard:
    """Monitors tokens/sec across calls to detect inference stalls or swap collapse."""

    def __init__(
        self,
        min_tokens_per_sec: float = DEFAULT_MIN_TOKENS_PER_SEC,
        streak_limit: int = DEFAULT_COLLAPSE_STREAK,
    ) -> None:
        self.min_tokens_per_sec = min_tokens_per_sec
        self.streak_limit = streak_limit
        self.consecutive_slow_calls = 0

    def record_call(self, tokens_per_sec: float) -> None:
        """Record call speed and raise ThroughputCollapsedError if threshold is violated repeatedly."""
        if tokens_per_sec > 0 and tokens_per_sec < self.min_tokens_per_sec:
            self.consecutive_slow_calls += 1
            logger.warning(
                "Slow inference detected: %.1f tokens/sec (< min %.1f). Consecutive: %d/%d",
                tokens_per_sec, self.min_tokens_per_sec, self.consecutive_slow_calls, self.streak_limit
            )
        else:
            self.consecutive_slow_calls = 0

        if self.consecutive_slow_calls >= self.streak_limit:
            msg = (
                f"Throughput collapsed: {self.consecutive_slow_calls} consecutive calls fell below "
                f"{self.min_tokens_per_sec:.1f} tokens/sec. Pausing run to protect system memory."
            )
            logger.error(msg)
            raise ThroughputCollapsedError(msg)


class CaffeinateContext:
    """Context manager wrapping long runs with macOS `caffeinate -i` to prevent sleep."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[Any] | None = None

    def __enter__(self) -> "CaffeinateContext":
        if os.uname().sysname == "Darwin":
            try:
                self.proc = subprocess.Popen(
                    ["caffeinate", "-i"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Started caffeinate -i (preventing macOS idle sleep)")
            except Exception as exc:
                logger.debug("Failed starting caffeinate: %s", exc)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
                logger.info("Terminated caffeinate")
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None


def check_battery_state() -> tuple[bool, str]:
    """Check if laptop is running on battery power via pmset."""
    if os.uname().sysname != "Darwin":
        return False, ""
    try:
        res = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False)
        if res.returncode == 0 and "Battery Power" in res.stdout:
            return True, res.stdout.strip()
    except Exception:
        pass
    return False, ""


def check_memory_pressure_critical() -> tuple[bool, str]:
    """Check if macOS memory pressure is critical via vm_stat."""
    if os.uname().sysname != "Darwin":
        return False, ""
    try:
        res = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            return False, ""
        free = 0
        speculative = 0
        for line in res.stdout.splitlines():
            if "Pages free:" in line:
                free = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages speculative:" in line:
                speculative = int(line.split(":")[1].strip().rstrip("."))
        free_mb = ((free + speculative) * 4096) / (1024 * 1024)
        if free_mb < 400:  # < 400 MB free RAM
            return True, f"Critical memory pressure: only {free_mb:.0f} MB free memory"
    except Exception:
        pass
    return False, ""

