---
allowed-tools: Bash(uv run ruff format:*), Bash(uv run ruff check:*), Bash(uv run pytest:*), Bash(uv run serve resolve:*)
description: Run the formatter, the linter, the tests, and resolve every preset
---

## Your task

Run the same checks as CI, then resolve every preset:

1. `uv run ruff format --check .`
2. `uv run ruff check .`
3. `uv run pytest -q`
4. `uv run serve resolve <preset>` for each file in `presets/`

Report one line per step with the result. If a step fails, show the failing output and the smallest
change that fixes it. Don't edit files unless the user asks.
