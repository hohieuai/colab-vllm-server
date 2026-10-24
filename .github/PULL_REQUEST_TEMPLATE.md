## What changed

<!-- One or two sentences. Link the issue it fixes, if any: Fixes #123 -->

## Why

<!-- The problem this solves. -->

## How it was tested

- [ ] `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` pass.
- [ ] For changes to `remote/launch.py`, the tunnel, or `serve check`: I ran
      `uv run serve check <preset> --accuracy` and pasted the summary below.
- [ ] For a new preset: I added its recipe files to `tests/fixtures/recipes/`.

<!-- Paste check output here. -->

## Checklist

- [ ] The title follows Conventional Commits, such as `feat: add serve chat`.
- [ ] `CHANGELOG.md` has an entry under *Unreleased* for any user-visible change.
- [ ] The documentation matches the change.
