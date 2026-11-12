"""
tests/devswarm/test_guards.py — Unit tests for loop detection, kill switch, and throughput collapse guard.
"""

import pytest
from devswarm.platform.guards import (
    LoopDetector,
    LoopDetectedError,
    KillSwitch,
    KillSwitchTriggered,
    ThroughputGuard,
    ThroughputCollapsedError,
    CaffeinateContext,
)


def test_loop_detector_triggers_on_third_duplicate():
    detector = LoopDetector(limit=3)

    # 1st call: ok
    detector.record_and_check("read_file", {"path": "calc.py"})
    # 2nd call: ok
    detector.record_and_check("read_file", {"path": "calc.py"})

    # 3rd identical call: MUST raise LoopDetectedError
    with pytest.raises(LoopDetectedError) as exc_info:
        detector.record_and_check("read_file", {"path": "calc.py"})

    assert "Loop detected" in str(exc_info.value)
    assert "read_file" in str(exc_info.value)


def test_loop_detector_resets_on_different_call():
    detector = LoopDetector(limit=3)

    detector.record_and_check("read_file", {"path": "calc.py"})
    detector.record_and_check("read_file", {"path": "calc.py"})
    # Intervening different call
    detector.record_and_check("list_dir", {"path": "."})
    # Reset sequence: 1st and 2nd should not raise
    detector.record_and_check("read_file", {"path": "calc.py"})
    detector.record_and_check("read_file", {"path": "calc.py"})


def test_kill_switch_triggers(tmp_path):
    stop_file = tmp_path / "STOP"
    kill_switch = KillSwitch(stop_file=stop_file)

    # Should not raise when file does not exist
    kill_switch.check()

    # Create STOP file
    kill_switch.trigger("Operator requested abort")

    with pytest.raises(KillSwitchTriggered) as exc_info:
        kill_switch.check()

    assert "Kill switch triggered" in str(exc_info.value)

    # Clear STOP file
    kill_switch.clear()
    kill_switch.check()  # Should not raise


def test_throughput_guard_triggers_after_consecutive_drops():
    guard = ThroughputGuard(min_tokens_per_sec=5.0, streak_limit=3)

    # Call 1: slow (3.0 tok/s)
    guard.record_call(3.0)
    assert guard.consecutive_slow_calls == 1

    # Call 2: slow (2.5 tok/s)
    guard.record_call(2.5)
    assert guard.consecutive_slow_calls == 2

    # Call 3: slow (1.8 tok/s) -> MUST raise ThroughputCollapsedError
    with pytest.raises(ThroughputCollapsedError) as exc_info:
        guard.record_call(1.8)

    assert "Throughput collapsed" in str(exc_info.value)


def test_throughput_guard_recovers_on_normal_speed():
    guard = ThroughputGuard(min_tokens_per_sec=5.0, streak_limit=3)

    guard.record_call(3.0)
    guard.record_call(2.5)
    # Fast call resets streak
    guard.record_call(25.0)
    assert guard.consecutive_slow_calls == 0


def test_caffeinate_context():
    # Verify context manager enters and exits cleanly
    with CaffeinateContext() as ctx:
        assert ctx is not None

