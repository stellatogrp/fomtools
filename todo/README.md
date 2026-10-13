# Documentation audit: TODO lists

This folder holds the findings of a full documentation audit of the repository, written as
task lists so that several people can work through them in parallel. The audit compared
every docstring under `fomtools/` with its signature and body, every tutorial notebook and
helper with the current API, and every hand-written markdown page (including `CLAUDE.md`
and the `skills/` folder) with `pyproject.toml`, the CI workflows and the code.

Baseline: all line numbers refer to commit `d235580` on the `documentation` branch. They
drift as files change; when a link lands in the wrong place, search for the quoted text or
the symbol name.

## Files

| File | Items | P0 | P1 | P2 | What is in it |
| --- | ---: | ---: | ---: | ---: | --- |
| [code-bugs.md](code-bugs.md) | 8 | 1 | 4 | 3 | code that contradicts its own docstring; fix code, docstring and a test together |
| [docstrings.md](docstrings.md) | 75 | 0 | 34 | 41 | docstrings under `fomtools/`, grouped by package and file |
| [markdown-docs.md](markdown-docs.md) | 62 | 3 | 29 | 30 | `docs/*.md`, `README.md`, `CONTRIBUTING.md`, `CLAUDE.md`, notation drift |
| [notebooks.md](notebooks.md) | 54 | 2 | 16 | 36 | tutorial notebooks, codegen write-ups, helpers, gallery pages, export script |
| [skills.md](skills.md) | 34 | 13 | 17 | 4 | `skills/` folder and its broken auto-sync workflow |
| [repo-hygiene.md](repo-hygiene.md) | 17 | 1 | 8 | 8 | API-page coverage gaps, nav, CI, ignore rules |
| total | 250 | 20 | 108 | 122 | |

## Priorities

- **P0**: breaks a user's first attempt or the CI build. Wrong install claims, examples that
  raise when copied, dependencies the docs build does not install.
- **P1**: a wrong statement a reader would act on. Stale defaults, nonexistent class names,
  behaviour described backwards, missing parameters on public API.
- **P2**: consistency and polish. Notation, wording, dead code, duplicated sections.

## Suggested order

1. [code-bugs.md](code-bugs.md) and the two P0 dependency items at the top of
   [notebooks.md](notebooks.md). These change behaviour or unblock the docs CI.
2. [markdown-docs.md](markdown-docs.md), `docs/installation.md` and `CLAUDE.md` first, then
   [skills.md](skills.md). These are the pages a new user or an agent reads first, and the
   skill sync has to work before its content is worth fixing.
3. [docstrings.md](docstrings.md), starting with `verifier.py` (the `Verifier` class has no
   docstring at all, so the API page renders it empty).
4. [notebooks.md](notebooks.md) prose and notation, and the API-page coverage gaps in
   [repo-hygiene.md](repo-hygiene.md).

## Conventions

- Claim an item by appending `(@yourname)` to its line. Tick it (`- [x]`) in the PR that
  fixes it; the lists are meant to be edited.
- One item is one change. Where the same stale wording appears in several files, the item
  lists every location so one PR can fix them all.
- Each item links to the file and line, states what is stale, and says what it should say.
  Facts the fixes rely on (the real defaults, extras, class names) are repeated at the top of
  each file so nobody has to re-derive them.
- The notation table in `CLAUDE.md` formally covers the tutorial notebooks only. Drift
  elsewhere in `docs/` is listed as an opportunity, not a violation; decide once whether
  the table applies to all of `docs/` and record it in `CLAUDE.md`.
- When every item in a file is done, delete the file. When this folder is empty, delete it.

## What the audit did not find

- No docstring describes a parameter that has been removed from its signature.
- No tutorial notebook or helper calls a fomtools function, method or keyword that no longer
  exists.
- Every `::: fomtools.…` directive on the API pages resolves, every link in the mkdocs nav
  exists, and the speed-up figures in the tutorial pages match the recorded timings.
