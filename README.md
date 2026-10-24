<div align="center">

# colab-vllm-server

Run an OpenAI-compatible vLLM endpoint for an open-weight model on a Google Colab GPU.

[![CI](https://github.com/hohieuai/colab-vllm-server/actions/workflows/ci.yml/badge.svg)](https://github.com/hohieuai/colab-vllm-server/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

</div>

## What it does

colab-vllm-server takes the server command for a model from [vLLM Recipes](https://recipes.vllm.ai),
starts [vLLM](https://github.com/vllm-project/vllm) on a Colab VM, and connects it to your computer
through an SSH tunnel. Any OpenAI SDK can then call it at `http://127.0.0.1:8000/v1`.

One command, `serve`, does each step:

- **resolve** prints the exact `vllm serve` command for a preset, and rejects a model that can't run
  on the chosen GPU before Colab charges anything.
- **up** creates the VM, installs vLLM, downloads the model, starts the server, and opens the tunnel.
- **chat** sends a message and prints the answer, as a quick test.
- **down** stops the server and releases the VM.

Each deployment is a preset in [`presets/`](presets), configured for the highest output quality:
full-precision or released-format weights, a BF16 KV cache, and the highest reasoning effort the
server can set. The server listens only on the VM's loopback interface, so your computer reaches it
through the tunnel and nothing else can.

> [!IMPORTANT]
> A running endpoint uses Colab compute units every hour until you run `serve down`, even when it
> receives no requests.

## Install and start

Before you start, you need:

- [uv](https://docs.astral.sh/uv/)
- The [`colab` CLI](https://pypi.org/project/google-colab-cli/) 0.7.5 or later, installed with
  `uv tool install google-colab-cli` and signed in once with `colab sessions`
- An `ed25519` SSH key in `~/.ssh`
- Colab compute units. The A100 80 GB used below is the High-RAM machine shape, which needs Colab
  Pro or Pro+.

1. Download colab-vllm-server:

   ```bash
   git clone https://github.com/hohieuai/colab-vllm-server
   cd colab-vllm-server
   ```

2. Preview the server command. This runs on your computer and costs nothing:

   ```bash
   uv run serve resolve qwen3.8-27b-a100-80g
   ```

   The output ends with the command that will run on the VM:

   ```
   vllm serve Qwen/Qwen3.8-27B --enable-auto-tool-choice --tool-call-parser qwen3_coder ...
   ```

3. Start the endpoint. The first start takes 15 to 20 minutes, most of it the model download:

   ```bash
   uv run serve up qwen3.8-27b-a100-80g
   ```

   When the endpoint is ready, the terminal prints:

   ```
   [serve] ready: Qwen/Qwen3.8-27B on Colab A100_80G
     export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
     export OPENAI_API_KEY=vllm
   ```

4. Send a message:

   ```bash
   uv run serve chat qwen3.8-27b-a100-80g 'Hello!'
   ```

   ```
   Hello! How can I help you today?
   [serve] Qwen/Qwen3.8-27B: 10 tokens in 1.1 s, finish_reason=stop
   ```

To stop the endpoint and release the VM, run `uv run serve down qwen3.8-27b-a100-80g`. The terminal
prints `[serve] qwen3.8-27b-a100-80g stopped and VM released`.

## Call it from your code

Load the endpoint's address and key into your shell:

```bash
eval "$(uv run serve env qwen3.8-27b-a100-80g)"
```

Then use any OpenAI SDK. The Python SDK reads both variables:

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

[`examples/curl`](examples/curl) and [`examples/python`](examples/python) hold one runnable example
for each API feature: streaming, reasoning, tool calls, structured outputs, and concurrent
requests. To keep the address and key in a file for scripts, copy `.env.example` to `.env.local`
and run `uv run serve env qwen3.8-27b-a100-80g --write`.

## Check from the terminal

`serve status` shows whether each endpoint answers:

```text
$ uv run serve status
qwen3.8-27b-a100-80g         Colab A100_80G session=vllm-qwen3-8-27b-a100-80g api=ready tunnel=up log=streaming http://127.0.0.1:8000/v1
```

If `api` is `unreachable` after your computer slept, run `uv run serve connect qwen3.8-27b-a100-80g`
to reopen the tunnel.

## Use it with an AI agent

colab-vllm-server ships an [agent skill](skills/colab-vllm-server/SKILL.md). With the skill
installed, ask your coding agent to "start a Qwen endpoint on Colab". The agent runs `serve up`,
sends a test message, and runs `serve down` when the work is done, so no VM keeps running by mistake.

To install it in Claude Code as a plugin, run:

```
/plugin install colab-vllm-server --marketplace hohieuai/colab-vllm-server
```

For another agent, copy `skills/colab-vllm-server/SKILL.md` into that agent's skills folder. The
skill works alongside the `colab-operator` skill of the Colab CLI and the
[Colab Monitor](https://github.com/hohieuai/colab-monitor) skill.

## Watch it with Colab Monitor

[Colab Monitor](https://github.com/hohieuai/colab-monitor) shows your Colab VMs, compute-unit
spending, and the server log in a local dashboard. colab-vllm-server writes the server log to
`~/colab-runs/`, where Colab Monitor reads it, so the log appears without extra setup:

```bash
uv tool install git+https://github.com/hohieuai/colab-monitor
colab-monitor --open
```

## Documentation

- The [quickstart](docs/quickstart.md) walks through the setup above with more explanation,
  starting from installing uv.
- The guides show how to [generate text](docs/guides/generate-text.md),
  [control reasoning](docs/guides/control-reasoning.md), [call functions](docs/guides/call-functions.md),
  [get structured outputs](docs/guides/get-structured-outputs.md),
  [run concurrent requests](docs/guides/run-concurrent-requests.md), and
  [add a deployment](docs/guides/add-a-deployment.md).
- The [command-line reference](docs/reference/command-line.md) lists every command and option, the
  files colab-vllm-server reads and writes, and its environment variables.
- The [API reference](docs/reference/api.md) lists the supported endpoints and parameters, the vLLM
  extensions, and where the server differs from the OpenAI API.
- The [preset](docs/reference/presets.md) and [GPU](docs/reference/gpus.md) references explain how a
  preset becomes a command and which Colab GPUs it can use.
- [How colab-vllm-server works](docs/explanation/how-it-works.md) explains the design: recipes,
  the tunnel, and the frontend choice.
- [Troubleshooting](docs/troubleshooting.md) covers connection errors, sign-in errors, a smaller GPU
  than expected, and truncated answers.
- The [security policy](SECURITY.md) explains how to report a vulnerability.
- The [contributing guide](CONTRIBUTING.md) explains how to set up a development copy, run the
  tests, and open a pull request.

## License

Copyright 2026 The colab-vllm-server Authors. colab-vllm-server is licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

colab-vllm-server is an unofficial community project and is not affiliated with Google Colab,
vLLM, or the providers of the models it serves.
