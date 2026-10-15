"""Wall-clock budget for one agent phase, enforced turn by turn through SDK hooks.

The in-process Claude SDK has no wall-clock limit of its own, only a turn cap,
and a single turn can run a long benchmark. A :class:`TimeBudget` instead checks
the clock around every tool call, through the SDK's public hook API
(``ClaudeAgentOptions(hooks=...)``):

* after each tool call (``PostToolUse``) the agent is told how much time is
  left, so it can pace itself;
* before each tool call (``PreToolUse``) a ``Bash`` command's timeout is capped
  at the time left, so no single command outlives the budget; once the budget is
  spent every tool is denied with an instruction to wrap up; and once a short
  grace period after that has also passed, the session is ended.

The hooks are plain async callables over a clock, so they can be exercised
directly without the SDK.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: Time the agent gets after the budget is spent to write its summary before the
#: session is ended outright.
GRACE_S = 60.0
#: The Bash tool's own ceiling on a command timeout, in milliseconds.
_BASH_MAX_TIMEOUT_MS = 600_000
#: Floor on a capped Bash timeout, so a command issued just before the deadline
#: can still start.
_BASH_MIN_TIMEOUT_MS = 1_000


def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


@dataclass
class TimeBudget:
    """A wall-clock budget of ``total_s`` seconds ending at ``deadline`` (``clock`` time)."""

    deadline: float
    total_s: float
    clock: Callable[[], float] = time.monotonic
    grace_s: float = GRACE_S
    #: Set once a hook has denied a tool or ended the session for lack of time.
    exhausted: bool = field(default=False, init=False)

    @classmethod
    def starting_now(
        cls, total_s: float, clock: Callable[[], float] = time.monotonic
    ) -> TimeBudget:
        return cls(deadline=clock() + total_s, total_s=total_s, clock=clock)

    def remaining(self) -> float:
        return self.deadline - self.clock()

    async def after_tool(self, input_data: Any, tool_use_id: str | None, context: Any) -> dict:
        """``PostToolUse``: tell the agent how much of its budget is left."""
        left = self.remaining()
        if left > 0:
            note = (
                f"[fomtools] Time budget: {_fmt(left)} of {_fmt(self.total_s)} remain. "
                "Pace your work so the solver passes verify and your summary is written "
                "before it runs out."
            )
        else:
            note = (
                "[fomtools] Time budget exhausted. Stop now: leave the solver in a state "
                "that passes verify and write your summary."
            )
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": note}}

    async def before_tool(self, input_data: Any, tool_use_id: str | None, context: Any) -> dict:
        """``PreToolUse``: cap Bash timeouts, deny tools once out of time, then end the session."""
        left = self.remaining()
        if left <= -self.grace_s:
            self.exhausted = True
            return {
                "continue_": False,
                "stopReason": f"fomtools: the {_fmt(self.total_s)} time budget is spent",
            }
        if left <= 0:
            self.exhausted = True
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "Time budget exhausted. Do not run more tools: stop and write your "
                        "summary now."
                    ),
                }
            }
        tool_input = dict((input_data or {}).get("tool_input") or {})
        if (input_data or {}).get("tool_name") == "Bash":
            cap_ms = max(_BASH_MIN_TIMEOUT_MS, min(_BASH_MAX_TIMEOUT_MS, int(left * 1000)))
            requested = tool_input.get("timeout")
            if requested is None or int(requested) > cap_ms:
                tool_input["timeout"] = cap_ms
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": tool_input,
                    }
                }
        return {}

    def hooks(self) -> dict[str, list]:
        """``ClaudeAgentOptions.hooks`` for this budget (every tool)."""
        from claude_agent_sdk import HookMatcher  # the caller already imported the SDK

        return {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[self.before_tool])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[self.after_tool])],
        }
