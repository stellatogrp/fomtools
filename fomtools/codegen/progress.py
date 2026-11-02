"""Terminal progress for `generate_code` (``verbose=True``).

Prints one line per phase, one line per code optimization round with a bar that
shrinks with the solve time, and a final summary: translated vs. optimized time,
the verification error, and the optimizations the agent kept. While an agent
runs on a terminal, a spinner shows the elapsed time; elsewhere (notebooks, logs)
a line marks the start of each phase. Standard library only; colors only on a
terminal and never with ``NO_COLOR`` set. Call `Progress.close` when the run ends,
also on error, to stop the spinner.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time

_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_VERIFY_ERR_RE = re.compile(r"worst max-abs error over (\d+) samples: (\S+)")


def verify_error(output: str | None) -> tuple[int, float] | None:
    """``(samples, worst max-abs error)`` from ``make verify`` output, if printed."""
    m = _VERIFY_ERR_RE.search(output or "")
    return (int(m[1]), float(m[2])) if m else None


def bar(fraction: float, width: int = 12) -> tuple[str, str]:
    """The filled and the empty part of a ``width``-character bar at ``fraction``."""
    filled = min(width, max(1, round(width * fraction)))
    return "█" * filled, "░" * (width - filled)


def _us(value: float) -> str:
    return f"{value:.3g} µs" if value < 1000 else f"{value:,.0f} µs"


class Progress:
    """Reporter for one code-generation run; every method is a no-op when disabled."""

    def __init__(self, enabled: bool = True, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.tty = enabled and hasattr(self.stream, "isatty") and self.stream.isatty()
        self.color = self.tty and not os.environ.get("NO_COLOR")
        self.t0: float | None = None  # translated solve time, in us
        self._label = ""
        self._start = 0.0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._active = False

    # -- output helpers -----------------------------------------------------
    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _emit(self, text: str) -> None:
        """Write ``text``, replacing what the stream's encoding cannot represent."""
        try:
            self.stream.write(text)
        except UnicodeEncodeError:
            enc = getattr(self.stream, "encoding", None) or "ascii"
            self.stream.write(text.encode(enc, "replace").decode(enc))
        self.stream.flush()

    def _write(self, line: str) -> None:
        self._emit(("\r\033[K" if self.tty else "") + line + "\n")

    def _stop_spinner(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    # -- phases -------------------------------------------------------------
    def start(self, label: str) -> None:
        """Begin a phase; on a terminal, spin with the elapsed time until `done`."""
        if not self.enabled:
            return
        self._stop_spinner()
        self._label, self._start, self._active = label, time.monotonic(), True
        if not self.tty:
            self._write(f"… {label}")
            return
        self._stop = threading.Event()

        def spin():
            i = 0
            while not self._stop.wait(0.1):
                elapsed = time.monotonic() - self._start
                frame = self._paint(_SPIN[i % len(_SPIN)], "36")
                self._emit(f"\r\033[K{frame} {label}  {self._paint(f'{elapsed:.0f} s', '2')}")
                i += 1

        self._thread = threading.Thread(target=spin, daemon=True)
        self._thread.start()

    def done(self, ok: bool = True, detail: str = "") -> None:
        """End the current phase with a ✓ or ✗ line."""
        if not self.enabled:
            return
        self._stop_spinner()
        self._active = False
        elapsed = time.monotonic() - self._start
        mark = self._paint("✓", "32") if ok else self._paint("✗", "31")
        text = f"{self._label:<28} {detail}".rstrip()
        self._write(f"{mark} {text}  {self._paint(f'{elapsed:.0f} s', '2')}")

    def close(self) -> None:
        """Stop the spinner; a phase still open (the run raised) is marked ✗."""
        if self.enabled and self._active:
            self.done(False, "stopped")

    def translated(self, ok: bool, verify_output: str | None, us: float | None) -> None:
        """End the translation phase: verification error and, if measured, its time."""
        if not self.enabled:
            return
        if not ok:
            self.done(False, "failed verification")
            return
        parts = [self._paint("VERIFY OK", "32")]
        err = verify_error(verify_output)
        if err:
            parts.append(f"max err {err[1]:.0e}")
        if us:
            parts.append(_us(us))
            self.t0 = us
        self.done(True, "  ".join(parts))

    def round(self, r) -> None:
        """End one code optimization round (an `OptimizationRound`)."""
        if not self.enabled:
            return
        if r.index == 1 and r.us_before:
            self.t0 = r.us_before  # the translated solver, timed head to head
        if not r.changed:
            self.done(True, self._paint("no further change", "2"))
        elif not r.correct:
            self.done(False, self._paint("failed verification, reverted", "31"))
        elif not r.kept:
            after = _us(r.us_after) if r.us_after else "not timed"
            before = f" vs {_us(r.us_before)} best" if r.us_before else ""
            self.done(False, f"{after}{before}, not faster, reverted")
        elif r.us_after is None:
            self.done(True, "VERIFY OK  (not timed)")
        else:
            self.t0 = self.t0 or r.us_before or r.us_after
            full, empty = bar(r.us_after / self.t0)
            detail = (
                f"{_us(self.t0)} → {_us(r.us_after)}  "
                f"{self._paint(full, '36')}{self._paint(empty, '2')}  "
                f"{self._paint(f'{self.t0 / r.us_after:.2f}x', '1')}  "
                f"{self._paint('VERIFY ✓', '32')}"
            )
            self.done(True, detail)

    # -- final summary ------------------------------------------------------
    def summary(
        self,
        name: str,
        timing,
        verify_output: str | None,
        applied: list[str],
        width: int = 30,
    ) -> None:
        """Bars for translated vs. optimized C, the NumPy time, and what the agent kept."""
        if not self.enabled:
            return
        lines = ["", self._paint(f"  {name}", "1")]
        t0 = timing.c_initial_us if timing else None
        t1 = timing.c_final_us if timing else None
        if t0 and t1:
            for label, us in (("translated C", t0), ("optimized C", t1)):
                code = "2" if us == t0 and t1 != t0 else "36"
                lines.append(
                    f"  {label:<13} {_us(us):>10}  {self._paint(bar(us / t0, width)[0], code)}"
                )
            gain = f"{t0 / t1:.2f}x faster" if t1 < t0 else f"{t0 / t1:.2f}x (within noise)"
            lines[-1] += "  " + self._paint(gain, "1;32" if t1 < t0 else "1")
        if timing and timing.python_us and t1:
            lines.append(self._paint(f"  {'NumPy':<13} {_us(timing.python_us):>10}", "2"))
        err = verify_error(verify_output)
        if err:
            lines.append(
                f"  {self._paint('VERIFY OK', '32')}  worst error {err[1]:.1e} "
                f"over {err[0]} samples"
            )
        if applied:
            lines.append(self._paint("  what the agent found:", "1"))
            lines += [f"  • {item[:72]}" for item in applied]
        for line in lines:
            self._write(line)
