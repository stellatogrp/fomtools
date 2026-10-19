The docs or source code in this repo have been updated. Your job is to
update the skills/fomtools/ directory so the installable skill stays in
sync with the current documentation and API.

Read the following sources to understand the current state:
1. docs/ directory (all .md files and tutorial .py files)
2. fomtools/verifier.py (Verifier class API)
3. fomtools/codegen/ (codegen API: generate.py, ir.py, config)
4. fomtools/cvxpygen/ (CVXPYgen integration)
5. AGENTS.md (project context for developers)

Then update these files to reflect the current docs/API:
- skills/fomtools/SKILL.md (main skill file)
- skills/fomtools/references/api.md (API reference)
- skills/fomtools/references/codegen.md (codegen reference)
- skills/fomtools/references/cvxpygen.md (CVXPYgen reference)

Rules:
- Keep the SKILL.md frontmatter format (name, description, license in YAML)
- The description field is the primary trigger mechanism -- keep it broad
  and include relevant keywords
- Keep SKILL.md under 500 lines; put details in references/
- Update code examples if the API changed
- Add new features/methods that were added to the docs
- Remove references to features that were removed
- Keep the same file structure and organization style

After updating, commit the changes with message:
"chore: sync fomtools skill with docs"
