"""Wall-clock budget for one agent phase.

`run_verify_loop` starts a `TimeBudget` for each phase and publishes it to the
driver as ``task.extra["budget"]``. `CliAgentDriver` gives every attempt what is
left of it as a hard process timeout, and tells the agent the deadline through
the ``FOMTOOLS_DEADLINE`` environment variable (Unix time), ``GRACE_S`` seconds
before that timeout. An agent CLI with tool hooks can then stop and write its
summary in time; the bundled ``claude`` spec does this with a ``PreToolUse``
hook.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

#: Time the agent gets after ``FOMTOOLS_DEADLINE`` to write its summary before the
#: process is ended.
GRACE_S = 60.0


@dataclass
class TimeBudget:
    """A wall-clock budget of ``total_s`` seconds ending at ``deadline`` (``clock`` time)."""

    deadline: float
    total_s: float
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def starting_now(
        cls, total_s: float, clock: Callable[[], float] = time.monotonic
    ) -> TimeBudget:
        return cls(deadline=clock() + total_s, total_s=total_s, clock=clock)

    def remaining(self) -> float:
        return self.deadline - self.clock()

    def agent_deadline(self) -> int:
        """The Unix time at which the agent should stop: ``GRACE_S`` before the end."""
        return int(time.time() + max(0.0, self.remaining() - GRACE_S))
