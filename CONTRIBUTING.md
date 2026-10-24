# Contributing to colab-vllm-server

Thanks for helping. Bug reports, fixes, new presets, and small focused features are welcome.

## Before you start

- For anything bigger than a small fix, open an issue first so we can agree on the approach.
- colab-vllm-server takes server settings from vLLM Recipes. A fix for a wrong vLLM flag usually belongs
  in the recipe, in [vllm-project/recipes](https://github.com/vllm-project/recipes), not here.
- Keep the runtime small. New runtime dependencies need a strong reason.

## Set up

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hohieuai/colab-vllm-server
cd colab-vllm-server
uv sync
uv run serve resolve qwen3.8-27b-a100-80g
```

The last command prints the server command for a preset. It needs no Colab account.

Starting an endpoint needs the `colab` CLI, a Colab account with compute units, and an SSH key.
See the [quickstart](docs/quickstart.md).

### Use the dev container

The repository includes a dev container with uv, the project environment, and the Colab CLI. It
copies your `~/.ssh` keys in with the permissions ssh requires and forwards port 8000 to your
computer. Open the folder in VS Code and choose **Reopen in Container**. Inside the container,
sign in to Colab once with `colab sessions`.

### Use Claude Code

`CLAUDE.md` gives Claude Code the project's commands and conventions. The repository adds three
commands:

| Command | Does |
|---|---|
| `/run-checks` | Runs the formatter, the linter, the tests, and resolves every preset. |
| `/add-preset <recipe> <gpu>` | Writes a preset for a model from recipes.vllm.ai and resolves it. |
| `/commit-push-pr` | Runs the checks, commits with a Conventional Commits message, and opens a pull request. |

## Checks

Run these before you open a pull request. CI runs the same commands on Python 3.11, 3.12, and
3.13:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

The unit tests in `tests/` run offline. They read recipe documents from `tests/fixtures/recipes/`
and fail if code tries to reach the network. To test against a recipe that has no fixture, copy
its files from `cache/recipes/` after a `serve resolve`.

Write the test first, watch it fail, then make it pass. Every bug fix needs a test that would have
caught it.

## Changes that need a live endpoint

The unit tests cover how presets resolve. They don't cover what runs on Colab. If your change
touches `remote/launch.py`, the tunnel, or `src/colab_vllm_server/check.py`, start an endpoint and run:

```bash
uv run serve check <preset> --accuracy
```

Paste the summary lines into the pull request.

## Add a preset

1. Follow [Add a deployment](docs/guides/add-a-deployment.md).
2. Set `frontend: python`.
3. Start the endpoint and run `serve check --accuracy`.
4. Add the recipe's files to `tests/fixtures/recipes/`, so the preset test covers it.
5. In the pull request, include the check summary and the `nvidia-smi` line from the server log.

## Code layout

| Path | Contents |
|---|---|
| `src/colab_vllm_server/recipes.py` | Fetches and caches documents from recipes.vllm.ai |
| `src/colab_vllm_server/resolve.py` | Builds the `vllm serve` command from a preset and runs the pre-flight checks |
| `src/colab_vllm_server/cli.py` | The `serve` command: VM lifecycle, tunnel, and state files |
| `src/colab_vllm_server/check.py` | The validation checks behind `serve check` |
| `remote/launch.py` | Runs inside the Colab kernel: `nvidia-smi`, install, and `vllm serve` |
| `evals/loadtest.py` | Concurrency sweep |

## Guidelines

- **Never put secrets in logs or reports.** The API key and `HF_TOKEN` stay in the state file and
  `.env.local`, both mode 600.
- **Keep the recipe's command.** Change the rendered argv only for a Colab constraint, and record
  every change as a resolver warning.
- **Fail before the VM starts.** If a problem can be found from the recipe, check it in
  `resolve.py`, not on the VM.
- **Keep documentation true.** A claim about server behavior needs a check or a measurement behind
  it.
- **Start every new source file with the license header.** Copy it from any `.py` file. The
  copyright line is `Copyright <year> The colab-vllm-server Authors`, with the year the file was
  created. `tests/test_license_headers.py` fails for a file without it.
- Update `CHANGELOG.md` under *Unreleased* for any user-visible change.

## Pull requests

Write the pull request title in [Conventional Commits](https://www.conventionalcommits.org/) form,
because the title becomes the commit message.

| Type | Use it for |
|---|---|
| `feat` | A new feature or preset |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `test` | Tests only |
| `refactor` | A code change that neither fixes a bug nor adds a feature |
| `build` | Packaging or dependencies |
| `ci` | CI configuration |

By contributing, you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE). Copyright in contributions stays with their authors and is
recorded collectively as "The colab-vllm-server Authors".
