# colab-vllm-server

[![CI](https://github.com/hohieuai/colab-vllm-server/actions/workflows/ci.yml/badge.svg)](https://github.com/hohieuai/colab-vllm-server/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Run an OpenAI-compatible endpoint for an open-weight model on a Google Colab GPU.
colab-vllm-server takes the server configuration from [vLLM Recipes](https://recipes.vllm.ai),
starts [vLLM](https://github.com/vllm-project/vllm) on a Colab VM, and connects it to your
computer through an SSH tunnel. Any OpenAI SDK can then call it.

```bash
uv run serve up qwen3.8-27b-a100-80g
eval "$(uv run serve env qwen3.8-27b-a100-80g)"
```

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## What it does

- **Serves the OpenAI API.** Chat Completions, Completions, streaming, function calling,
  structured outputs, and `n` samples per request.
- **Uses the recipe's command.** A preset names a recipe, a checkpoint variant, and a GPU. The
  `vllm serve` command comes from the rendering that vLLM Recipes publishes for that pair.
- **Rejects what can't run.** A preset whose model does not fit the GPU fails before Colab charges
  compute units.
- **Keeps the server private.** The server listens only on the VM's loopback interface. Your
  computer reaches it through the tunnel.
- **Tests the deployment.** `serve check` tests what experiments depend on: reasoning parsing,
  tool calls, schema enforcement, determinism, long context, and accuracy.
- **Records each run.** Every deployment saves the resolved command, the recipe commit, the vLLM
  version, and the `nvidia-smi` output.

## Requirements

- macOS or Linux, with `git` and OpenSSH. The Colab CLI does not support Windows.
- [uv](https://docs.astral.sh/uv/). uv installs the Python versions the tools need.
- The [`colab` CLI](https://pypi.org/project/google-colab-cli/) 0.7.5 or later, signed in once
  with `colab sessions`.
- Colab compute units, from a paid plan or a purchase, and access to the GPU you choose. GPU
  availability varies by plan. The High-RAM machine shape, which the `A100_80G` GPU uses, needs
  Colab Pro or Pro+.
- An `ed25519` or `ecdsa` SSH key in `~/.ssh`.

## Get started

These steps take you from a computer without any of the tools to a first request to Qwen3.8-27B
on a Colab A100 80 GB, and back to a stopped VM. They take about 20 minutes, most of it the first
download of the model. Run each command in a terminal. The output shown under a step is what you
should see.

1. Install [uv](https://docs.astral.sh/uv/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Open a new terminal, then check that uv is on your `PATH`:

   ```bash
   uv --version
   ```

   ```
   uv 0.11.21
   ```

   Your version number can differ. Any recent version works. You don't need to install Python, because uv installs the version
   each tool needs.

2. Install the Colab CLI, version 0.7.5 or later:

   ```bash
   uv tool install google-colab-cli
   colab version
   ```

   ```
   Version: 0.7.5
   ```

3. Sign in to Colab. The command prints a URL. Open it in a browser, approve access, and paste the
   code back into the terminal:

   ```bash
   colab sessions
   ```

   After the sign-in, the command lists your Colab VMs. With none running, it prints:

   ```
   [colab] No active sessions found on server.
   ```

4. Check that you have an SSH key. The tunnel to the VM signs in with it:

   ```bash
   ls ~/.ssh/id_ed25519.pub
   ```

   If the command prints `No such file or directory`, create a key and press Enter at each prompt:

   ```bash
   ssh-keygen -t ed25519
   ```

5. Download colab-vllm-server and go into its folder. Run every later command from this folder:

   ```bash
   git clone https://github.com/hohieuai/colab-vllm-server
   cd colab-vllm-server
   ```

6. Preview the server command. This step runs on your computer and costs nothing. The first run
   also installs the project's dependencies:

   ```bash
   uv run serve resolve qwen3.8-27b-a100-80g
   ```

   The output ends with the command that will run on the VM:

   ```
   vllm serve Qwen/Qwen3.8-27B --enable-auto-tool-choice --tool-call-parser qwen3_coder ...
   ```

7. Start the endpoint. From this step on, Colab charges about 7 compute units per hour until
   step 9:

   ```bash
   uv run serve up qwen3.8-27b-a100-80g
   ```

   The command creates the VM, installs vLLM, downloads the model, starts the server, and opens
   the tunnel. This takes 15 to 20 minutes. To watch the progress, run
   `tail -f ~/colab-runs/vllm-qwen3-8-27b-a100-80g.log` in a second terminal. When the endpoint is
   ready, the command prints:

   ```
   [serve] ready: Qwen/Qwen3.8-27B on Colab A100_80G
     export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
     export OPENAI_API_KEY=vllm
   ```

8. Send a first message. `serve chat` sends it to the endpoint and prints the answer as it
   arrives:

   ```bash
   uv run serve chat qwen3.8-27b-a100-80g 'Hello!'
   ```

   The model replies. The wording differs from run to run. A last line reports the token count and
   the time:

   ```
   Hello! How can I help you today?
   [serve] Qwen/Qwen3.8-27B: 10 tokens in 1.1 s, finish_reason=stop
   ```

   To call the endpoint as any OpenAI client would, load its address and key into your shell and
   run the curl example:

   ```bash
   eval "$(uv run serve env qwen3.8-27b-a100-80g)"
   bash examples/curl/chat_completions.sh
   ```

   The server answers with JSON. The answer is in `choices[0].message.content`:

   ```
   "content": "Hello! How can I help you today?"
   ```

   To send the same question from Python with the OpenAI SDK, run the Python example. It prints
   the answer and the token counts:

   ```bash
   uv run --with openai python examples/python/chat_completions.py
   ```

9. Stop the endpoint, so that Colab stops charging compute units:

   ```bash
   uv run serve down qwen3.8-27b-a100-80g
   ```

   ```
   [serve] qwen3.8-27b-a100-80g stopped and VM released
   ```

To check that no VM is left running, run `colab sessions`. It prints
`No active sessions found on server.`

The [quickstart](docs/quickstart.md) walks through the same steps with more explanation.

### Use the endpoint day to day

| To | Run |
|---|---|
| Start the endpoint again | `uv run serve up qwen3.8-27b-a100-80g` |
| Send a test message | `uv run serve chat qwen3.8-27b-a100-80g 'Hello!'` |
| Load the address and key into a new terminal | `eval "$(uv run serve env qwen3.8-27b-a100-80g)"` |
| Save the address and key for scripts | `cp .env.example .env.local` then `uv run serve env qwen3.8-27b-a100-80g --write` |
| See whether the endpoint answers | `uv run serve status` |
| Reconnect after your computer slept | `uv run serve connect qwen3.8-27b-a100-80g` |
| Test the endpoint before an experiment | `uv run serve check qwen3.8-27b-a100-80g` |
| Stop it | `uv run serve down qwen3.8-27b-a100-80g` |

If a step fails, see [Troubleshooting](docs/troubleshooting.md).

### Use it from Claude Code

The repository includes a Claude Code plugin that teaches Claude Code to start, test, and stop
endpoints, and to stop the VM when the work is done. Install it in Claude Code:

```
/plugin install colab-vllm-server --marketplace hohieuai/colab-vllm-server
```

### Monitor with Colab Monitor

[Colab Monitor](https://github.com/hohieuai/colab-monitor) shows your Colab VMs, compute-unit
spending, and the server log in a local dashboard. colab-vllm-server writes the server log where
Colab Monitor reads it, so the log appears without extra setup:

```bash
uv tool install git+https://github.com/hohieuai/colab-monitor
colab-monitor --open
```

## Documentation

| Page | Contents |
|---|---|
| [Quickstart](docs/quickstart.md) | Start an endpoint and send a first request. |
| **Guides** | |
| [Generate text](docs/guides/generate-text.md) | Send chat requests, continue conversations, and stream responses. |
| [Control reasoning](docs/guides/control-reasoning.md) | Read the reasoning, set the effort, or turn thinking off. |
| [Call functions](docs/guides/call-functions.md) | Let the model call your tools. |
| [Get structured outputs](docs/guides/get-structured-outputs.md) | Get output that matches a JSON Schema. |
| [Get reproducible samples](docs/guides/get-reproducible-samples.md) | Sample so that a rerun gives the same outputs. |
| [Run concurrent requests](docs/guides/run-concurrent-requests.md) | Send many requests at once and measure capacity. |
| [Add a deployment](docs/guides/add-a-deployment.md) | Serve another model or use another GPU. |
| [Validate a deployment](docs/guides/validate-a-deployment.md) | Test an endpoint before an experiment. |
| [Troubleshooting](docs/troubleshooting.md) | Fix common errors. |
| **Reference** | |
| [Command line](docs/reference/command-line.md) | `serve` commands, options, files, and variables. |
| [API](docs/reference/api.md) | Endpoints, parameters, vLLM extensions, and differences from OpenAI. |
| [Presets](docs/reference/presets.md) | Preset fields, resolution order, and pre-flight checks. |
| [GPUs](docs/reference/gpus.md) | Colab GPUs and the recipe profiles they use. |
| [Validation results](docs/reference/validation-results.md) | Checks and measured results. |
| **Explanation** | |
| [How colab-vllm-server works](docs/explanation/how-it-works.md) | Recipes, the tunnel, and the frontend choice. |
| [Reproducibility](docs/explanation/reproducibility.md) | Why some outputs change between runs. |

The [`examples/curl`](examples/curl) and [`examples/python`](examples/python) folders hold one
runnable example per API feature.

## Project layout

| Path | Contents |
|---|---|
| `colab_gpus.yaml` | Colab GPUs and the recipe profile each one uses |
| `presets/` | One YAML file per deployment |
| `src/colab_vllm_server/` | Resolver, command line, and validation suite |
| `remote/launch.py` | Script that runs on the VM: `nvidia-smi`, install, `vllm serve` |
| `examples/` | curl and Python examples |
| `evals/loadtest.py` | Concurrency sweep |
| `tests/` | Offline unit tests |
| `plugins/colab-vllm-server/` | Claude Code plugin, listed in `.claude-plugin/marketplace.json` |
| `.claude/commands/` | Claude Code commands for contributors: `/run-checks`, `/add-preset`, `/commit-push-pr` |
| `.devcontainer/` | Dev container with uv, the Colab CLI, and your SSH keys |
| `.github/` | CI, the `@claude` workflow, issue and pull request templates, Dependabot |
| `.vscode/` | Recommended extensions and settings |
| `docs/` | Documentation |

## Presets

Every preset is configured for the highest output quality: full-precision or released-format
weights, a BF16 KV cache, and the highest reasoning effort the server can set.

| Preset | Model | Colab GPU | Status |
|---|---|---|---|
| `qwen3.8-27b-a100-80g` | Qwen3.8-27B, BF16 | A100 80 GB | Validated end to end with vLLM 0.32.0, October 2026 |
| `gpt-oss-120b-g4` | gpt-oss-120b, MXFP4 | G4 (RTX PRO 6000) | Resolves. Not run on Colab yet |
| `gpt-oss-120b-h100` | gpt-oss-120b, MXFP4 | H100 | Resolves. Not run on Colab yet |
| `llama-3.1-8b-instruct-a100` | Llama 3.1 8B Instruct, BF16 | A100 40 GB | Resolves. Not run on Colab yet. Needs `HF_TOKEN` |

Run `serve check` on a deployment before you use it for experiments.

## Contributing

Bug reports and focused fixes are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first. To
report a security problem, follow [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
