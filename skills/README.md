# FOMTools Agent Skill

An installable [Agent Skill](https://agentskills.io/) that gives AI coding
agents (Claude Code, Cursor, Codex, etc.) knowledge of the FOMTools API,
verification pipeline, and code generation system.

The skill uses the open `SKILL.md` format and contains no harness-specific
instructions or tools.

## Install

The [`skills` CLI](https://github.com/vercel-labs/skills) requires Node.js and
installs skills into supported AI agents. If `npx` is unavailable, install the
[LTS release of Node.js](https://nodejs.org/en/download), open a new terminal,
and try again.

To inspect the available skills before installing them, run:

```bash
npx skills add stellatogrp/algo_verify --list
```

Install the skill globally so it is available across projects:

```bash
npx skills add stellatogrp/algo_verify -g
```

The installer detects supported agents and asks where to install the skill.

As with any public skill, read its `SKILL.md` before installing it.

## Update

Update the skill to the latest version with:

```bash
npx skills@latest update fomtools -g
```

Start a new agent session after updating so it loads the new instructions.

## What it provides

- FOMTools API reference (Verifier, LinearExpression, steps, constraints)
- Verification pipeline guide (MIP, SDP, simulator methods)
- Code generation reference (embedded C, OpenMP, CUDA targets)
- CVXPYgen integration docs
- Common usage patterns and examples

## Auto-updating

The skill is automatically kept in sync with the docs via the
`.github/workflows/update-skill.yml` CI workflow. When docs or source code
change on `main`, Claude Code updates the skill files and commits the changes.
