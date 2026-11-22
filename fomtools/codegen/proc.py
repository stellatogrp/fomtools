"""Running the package's own checks (build, verify, bench, profile) with a time limit.

These commands build and run agent-written code, which may never return. Each
runs in its own session, so at the limit the whole process group is ended:
``make``, the compiler it started, and the solver binary.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

#: Default limit, in seconds, of one check (`CodegenConfig.check_timeout_s`).
DEFAULT_CHECK_TIMEOUT_S = 1800

#: Exit status reported for a check that was ended at its limit (as ``timeout(1)``).
TIMEOUT_RETURNCODE = 124


def decode_stream(stream) -> str:
    """Decode a captured subprocess stream that may be bytes, str, or absent.

    ``subprocess`` populates ``TimeoutExpired.stdout`` with whatever it
    captured before the timeout, but skips the text decoding it would have
    applied on the normal path — so the timeout branch sees ``bytes`` even under
    ``text=True``.
    """
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


def popen_in_session(cmd: list[str], **kwargs) -> subprocess.Popen:
    """Start ``cmd`` in its own session, with text pipes for its output.

    Its own session, so `kill_group` ends it together with what it started. The
    child inherits this thread's signal mask, and some imports block SIGINT and
    SIGTERM (``import cvxpygen`` does), so it is started with an empty mask and
    can still be interrupted and asked to stop.
    """
    unmask = hasattr(signal, "pthread_sigmask")
    mask = signal.pthread_sigmask(signal.SIG_SETMASK, ()) if unmask else None
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",  # a tool may print bytes that are not valid text
            start_new_session=True,
            **kwargs,
        )
    finally:
        if unmask:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def kill_group(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    """End ``proc`` and every process in its session (it was started in its own).

    SIGTERM first so the process can stop what it started, then SIGKILL for whatever
    is still in the group after ``grace_s``. Also cleans up after a process that
    already exited, which leaves only its leftover children in the group.
    """
    if not hasattr(os, "killpg"):  # no process groups (Windows): end the process itself
        proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return  # the group is empty
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            proc.wait(timeout=0.05)  # reap the process so it does not hold the group
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_check(
    cmd: list[str], cwd: str | Path, timeout_s: float | None = DEFAULT_CHECK_TIMEOUT_S, **kwargs
) -> subprocess.CompletedProcess:
    """Run ``cmd`` in ``cwd`` and capture its output, for at most ``timeout_s`` seconds.

    A command still running at the limit is ended together with everything it
    started. That is reported as a failure, not raised: the result has the
    nonzero ``returncode`` `TIMEOUT_RETURNCODE`, the output captured so far, and
    a note at the end of ``stderr`` saying the check was terminated. ``None``
    disables the limit. ``kwargs`` go to ``subprocess.Popen``.
    """
    proc = popen_in_session(cmd, cwd=str(cwd), **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        kill_group(proc, grace_s=1.0)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired as late:  # a detached child still holds the pipes
            exc = late if late.stdout or late.stderr else exc
            stdout, stderr = decode_stream(exc.stdout), decode_stream(exc.stderr)
        note = (
            f"\n[fomtools: `{shlex.join(map(str, cmd))}` exceeded its {timeout_s:g}s limit and "
            "was terminated, probably because of a loop that does not terminate; any output "
            "above is partial.]\n"
        )
        return subprocess.CompletedProcess(cmd, TIMEOUT_RETURNCODE, stdout, stderr + note)
    except BaseException:  # e.g. KeyboardInterrupt: do not leave the check running
        kill_group(proc, grace_s=1.0)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
