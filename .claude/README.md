# `.claude/` — Claude Code configuration

This directory holds shared Claude Code settings for the project. It is checked
into version control so the whole team shares conventions.

## Files

- **`settings.json`** — shared, committed settings. Currently it only adds
  protective `deny` rules so secrets (`.env`) are never read. No Bash permissions
  are auto-granted; add them deliberately (see below).
- **`settings.local.json`** — *git-ignored*, per-developer overrides. Create your
  own; it won't be committed.

## Recommended permission allowlist (opt in deliberately)

To cut repetitive approval prompts for the safe, common commands in this project,
add an allowlist. **Do this via the `/permissions` command** (or the
`/fewer-permission-prompts` skill) so each member approves it consciously rather
than inheriting it. Suggested rules:

```json
"permissions": {
  "allow": [
    "Bash(python:*)",
    "Bash(python3:*)",
    "Bash(pip:*)",
    "Bash(pytest:*)",
    "Bash(ruff:*)",
    "Bash(latexmk:*)",
    "Bash(pdflatex:*)",
    "Bash(git status)",
    "Bash(git diff:*)",
    "Bash(git log:*)"
  ]
}
```

## Conventions for working with Claude here

- Read [`../CLAUDE.md`](../CLAUDE.md) first — it defines the load-bearing
  conventions (validation split, letter-based answers, deterministic decoding +
  manifests) that must not be silently changed.
- Keep `CLAUDE.md` updated as the "Decisions pending" list gets resolved.
