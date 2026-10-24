---
allowed-tools: Bash(git checkout -b:*), Bash(git add:*), Bash(git status:*), Bash(git diff:*), Bash(git push:*), Bash(git commit:*), Bash(gh pr create:*), Bash(uv run ruff:*), Bash(uv run pytest:*)
description: Run the checks, commit, push, and open a pull request
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`

## Your task

Based on the changes above:

1. Run `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest -q`. If any fails,
   stop and report it.
2. If the branch is `main`, create a new branch named after the change.
3. Create one commit. Write the message in Conventional Commits form, such as
   `feat: add serve chat` or `docs: fix quickstart step 3`.
4. If the change is user-visible and `CHANGELOG.md` has no entry for it, stop and say so.
5. Push the branch and open a pull request with `gh pr create`. Write a body of a few lines: what
   changed, why, and how it was tested.
