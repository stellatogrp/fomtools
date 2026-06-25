"""Pi (https://pi.dev) coding-agent backend.

This module supplies the Pi-specific pieces of a real, nondeterministic coding
agent driver: the ``pi`` argument vector and the multi-provider environment
policy. The shared retry/verify machinery lives in the base class.

Pi ships no Python SDK (its SDK is Node-only), so the in-process SDK path is not
implemented and every attempt runs through the headless ``pi --print`` CLI.
After each attempt the shared loop runs the verify command itself to decide
success, never the agent's self-report.

Pi is multi-provider (Anthropic/OpenAI/Google via OAuth or whichever provider
key is already in the environment), so — unlike the single-key Codex/Gemini
backends — no API key is forwarded: the child inherits the environment unchanged
and Pi resolves its own credentials. Like every real backend it needs
authentication (a logged-in ``pi`` via ``/login``, or a provider key such as
``ANTHROPIC_API_KEY``) and so never runs in CI; the deterministic pipeline is
exercised via :class:`~fomtools.codegen.agent.stub.StubAgentDriver`.
"""

from __future__ import annotations


from fomtools.codegen.agent.driver import AgentTask
from fomtools.codegen.agent.sdk_base import _SYSTEM_PROMPT, SdkAgentDriver


class PiDevAgentDriver(SdkAgentDriver):
    """Drive code generation with the Pi (pi.dev) agent.

    A concrete :class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver` that runs
    a real Pi coding agent through the authoritative retry loop. Pi has no Python
    SDK, so the in-process SDK path is unimplemented and every attempt is served
    by the headless ``pi --print`` CLI. Pi loops internally until the run ends,
    so the ``max_turns`` budget is accepted for interface parity but unused.

    Methods
    -------
    run
        Generate/refine the solver source until the verify command passes
        (inherited from :class:`~fomtools.codegen.agent.sdk_base.SdkAgentDriver`).

    Attributes
    ----------
    api_key_env : str
        Environment variable consulted for the auth-failure hint
        (``"ANTHROPIC_API_KEY"``); no key is forced onto the child, since Pi is
        multi-provider.
    cli_name : str
        CLI command name (``"pi"``), used in auth-failure messages.
    label : str
        Human-readable backend name (``"Pi"``), used in auth-failure messages.
    """

    # Pi is multi-provider, so there is no single key to force; api_key stays None
    # by default (see _child_env) and the child inherits the parent environment.
    # api_key_env feeds only the auth-failure hint built by the base class.
    api_key_env = "ANTHROPIC_API_KEY"
    cli_name = "pi"
    label = "Pi"

    def _child_env(self) -> dict[str, str] | None:
        """Inherit the parent environment unchanged so Pi resolves its own auth.

        Pi is multi-provider (OAuth login or any of ``ANTHROPIC_API_KEY`` /
        ``OPENAI_API_KEY`` / ``GOOGLE_API_KEY`` already in the environment).
        Unlike the single-key backends there is no one key to force, so we never
        rewrite the environment: returning ``None`` makes the subprocess inherit
        the parent environment, where Pi finds whichever credential is present.

        Returns
        -------
        None
            Always, so the child process inherits the parent environment.
        """
        return None

    def _build_argv(self, workdir: str, prompt: str) -> list[str]:
        """Assemble the ``pi --print`` argument vector for one headless attempt.

        Builds a non-interactive ``pi`` invocation. ``--print`` prints the
        response and exits instead of launching the TUI; ``--approve`` trusts the
        local files so the agent can apply edits and run ``make`` unattended in
        its isolated output dir. ``--model`` is added only when a model is
        configured. Pi has no working-directory flag, so ``workdir`` is scoped via
        the subprocess cwd rather than an argv flag, and no system-prompt flag, so
        the shared ``_SYSTEM_PROMPT`` framing is prepended to the prompt, which is
        passed positionally (last).

        Parameters
        ----------
        workdir : str
            Absolute working directory. Accepted for interface parity with the
            other backends; Pi takes no cwd flag, so it is applied via the
            subprocess cwd in :meth:`_invoke_cli` instead of the argv.
        prompt : str
            The task prompt for this attempt.

        Returns
        -------
        list of str
            The argument vector to pass to :func:`subprocess.run`.
        """
        argv = ["pi", "--print", "--approve"]
        if self.model:
            argv += ["--model", self.model]
        argv.append(_SYSTEM_PROMPT + "\n\n" + prompt)
        return argv

    def _invoke_sdk(self, task: AgentTask, prompt: str, max_turns: int) -> str:
        """Reject the in-process SDK path, since Pi ships no Python SDK.

        Pi's SDK is Node-only, so there is no in-process path to drive. This
        override exists only to satisfy the abstract method; the backend always
        runs through the ``pi --print`` CLI built by :meth:`_build_argv`.

        Parameters
        ----------
        task : AgentTask
            The unit of work. Unused; accepted for interface parity with the base
            class.
        prompt : str
            The task prompt for this attempt. Unused.
        max_turns : int
            Per-attempt turn budget. Unused.

        Raises
        ------
        NotImplementedError
            Always, because Pi exposes no in-process Python SDK.
        """
        raise NotImplementedError()

 