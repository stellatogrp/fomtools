"""A deterministic, LLM-free agent driver for testing.

The stub drops a pre-written ("golden") solver source into the workspace and
runs the verify command, so the entire code-generation pipeline can be exercised
in CI without any model calls or network access.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fomtools.codegen.agent.driver import AgentDriver, AgentResult, AgentTask


class StubAgentDriver(AgentDriver):
    """A deterministic :class:`AgentDriver` that writes a fixed solver source.

    Instead of calling an LLM, it writes a pre-supplied ("golden") solver body
    into the workspace and runs the real verify command, letting the full
    code-generation pipeline be exercised in CI without a model or network.

    Methods
    -------
    run
        Write the fixed source into the task workspace and verify it.

    """

    def __init__(self, solver_source: str | Path):
        """Resolve and store the solver source to write on every :meth:`run`.

        Parameters
        ----------
        solver_source : str or Path
            The golden solver body. Either C source text, or a path (``Path``
            or ``str``) to a ``.c`` file whose contents are read in. A ``str``
            is treated as a path only when it has no newlines, is under 4096
            characters, and names an existing file; otherwise it is taken as
            source text verbatim.
        """
        if isinstance(solver_source, Path):
            self._source = solver_source.read_text()
            return
        # A str argument may be either a path to a .c file or the source itself.
        # Guard the filesystem probe: long/multi-line source confuses Path.
        self._source = solver_source
        if "\n" not in solver_source and len(solver_source) < 4096:
            try:
                path = Path(solver_source)
                if path.is_file():
                    self._source = path.read_text()
            except OSError:
                pass

    def run(self, task: AgentTask) -> AgentResult:
        """Write the fixed source into the workspace and run the verify command.

        Implements the :class:`AgentDriver` interface. The stored source is
        written to ``task.solver_rel_path``, ``task.restore_hook`` is invoked (if present) to
        restore the deterministic files before verifying; then ``task.verify_cmd`` is run in
        ``task.workdir`` and its exit code decides success.

        Parameters
        ----------
        task : AgentTask
            The unit of work describing the workspace, the solver path to
            write, the restore hook, and the verify command to run.

        Returns
        -------
        AgentResult
            Result whose ``success`` reflects the verify command's exit code,
            with the combined stdout/stderr in ``log``, ``turns_used`` always
            1, and the written file in ``artifacts``.
        """
        target = task.workdir / task.solver_rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._source)

        if task.restore_hook is not None:
            task.restore_hook()
        proc = subprocess.run(
            task.verify_cmd,
            cwd=task.workdir,
            capture_output=True,
            text=True,
        )
        log = proc.stdout + proc.stderr
        return AgentResult(success=proc.returncode == 0, log=log, turns_used=1, artifacts=[target])
