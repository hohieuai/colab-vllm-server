# colab-vllm-server

Runs an OpenAI-compatible vLLM endpoint on a Google Colab GPU. The `serve` command resolves a
preset into a `vllm serve` command from recipes.vllm.ai, starts it on a Colab VM, and opens an SSH
tunnel to it.

## Commands

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
uv run serve resolve <preset>
```

`serve resolve` and the tests run offline. `serve up`, `serve chat`, `serve check`, and
`serve down` use a Colab VM and cost compute units. Don't run `serve up` unless the user asks, and
run `serve down` when the work is done.

## Layout

| Path | Contents |
|---|---|
| `src/colab_vllm_server/resolve.py` | Preset to command, pre-flight checks, borrowed-profile adaptation |
| `src/colab_vllm_server/cli.py` | The `serve` command: VM lifecycle, tunnel, state files |
| `src/colab_vllm_server/check.py` | Validation checks behind `serve check` |
| `src/colab_vllm_server/recipes.py` | Fetches and caches recipes.vllm.ai documents |
| `remote/launch.py` | Runs inside the Colab kernel: `nvidia-smi`, install, `vllm serve` |
| `presets/`, `colab_gpus.yaml` | Deployments and Colab GPUs |
| `tests/fixtures/recipes/` | Saved recipe documents for offline tests |
| `docs/` | Documentation, organized as tutorial, guides, reference, and explanation |

## Conventions

- Every `.py` and `.sh` file starts with the Apache 2.0 header, copyright
  `The colab-vllm-server Authors`. `tests/test_license_headers.py` enforces it.
- Tests are offline. Add the recipe files a test needs to `tests/fixtures/recipes/`.
- Keep the recipe's rendered command. Change it only for a Colab constraint, and report every change
  as a resolver warning.
- Every claim in `docs/` about server behavior needs a test or a measurement.
- Docs follow Google developer style: second person, present tense, sentence-case headings, no
  semicolons or em dashes in prose. Shell examples have no trailing comments, and chat messages use
  single quotes, because zsh is the default shell on macOS.
- Pull request titles use Conventional Commits. Update `CHANGELOG.md` under *Unreleased* for
  user-visible changes.

## Colab pitfalls

- A running endpoint occupies the notebook kernel. `colab exec` on that VM never returns. Use
  `colab console -s <session>` for shell commands.
- A VM accepts one `colab ssh` connection, and the tunnel holds it.
- `colab new --gpu` silently falls back to an A100 for an unknown value. The resolver validates it.
