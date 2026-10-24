# Command-line reference

## Synopsis

```
uv run serve <command> [options]
```

Every command that takes `<preset>` accepts a preset name, which is a file in `presets/` without
`.yaml`, or a path to a preset file.

## Commands

| Command | Does |
|---|---|
| `serve resolve <preset>` | Prints the resolved `vllm serve` command, install script, and warnings. Runs offline. |
| `serve up <preset>` | Creates the VM, installs vLLM, starts the server, and opens the tunnel. |
| `serve env <preset>` | Prints the client variables for a running endpoint. |
| `serve chat <preset> [message]` | Sends one chat message to a running endpoint and prints the answer. |
| `serve check <preset>` | Runs the validation checks against a running endpoint. |
| `serve connect <preset>` | Reopens the SSH tunnel to a running endpoint. |
| `serve status` | Lists endpoints started from this checkout. |
| `serve down <preset>` | Stops the tunnel and the server and releases the VM. |

## serve resolve

```
uv run serve resolve <preset> [--json] [--refresh]
```

| Option | Description |
|---|---|
| `--json` | Print the full resolved configuration as JSON. |
| `--refresh` | Fetch the recipe from recipes.vllm.ai instead of the cache. |

Exits with status 1 and a one-line reason when a [pre-flight check](presets.md#pre-flight-checks)
fails.

## serve up

```
uv run serve up <preset> [--port PORT] [--restart] [--high-mem] [--refresh]
```

| Option | Description |
|---|---|
| `--port PORT` | Local port for the tunnel. Default: the endpoint's previous port, or the first free port from 8000. A port in use stops the command before a VM starts. |
| `--restart` | Relaunch the server on the preset's existing VM. Restarts the kernel and stops the old server. The installed environment is kept when the install script has not changed since the last finished install. |
| `--high-mem` | Request the High-RAM machine shape. GPUs with `high_mem: true` request it without this option. |
| `--refresh` | Fetch the recipe again before resolving. |

Before anything else, `up` runs `colab sessions` to confirm that the colab CLI is signed in. If a
VM for the preset exists, `up` stops and asks for `--restart`, because the server on that VM may be
healthy behind a dropped tunnel. Run `serve connect` for that case. `up` waits up to 45 minutes for
`GET /v1/models` to list the preset's model, and reconnects a dropped tunnel up to 5 times. The
launch stops if Colab assigns a GPU with less memory than the preset expects.

## serve env

```
uv run serve env <preset> [--write] [--dotenv]
```

Without options, prints shell exports:

```
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=vllm
```

| Option | Description |
|---|---|
| `--write` | Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in `.env.local` and keep every other line. Creates the file with mode 600. |
| `--dotenv` | Print `KEY=value` lines without `export`. |

## serve chat

```
uv run serve chat <preset> [message] [--think] [--max-tokens N]
```

Sends one user message to the endpoint, streams the answer to standard output, and prints a
summary line with the token count, the time, and `finish_reason` to standard error. Without
`message`, sends `Hello! Introduce yourself in one sentence.`

| Option | Description |
|---|---|
| `--think` | Turn the model's thinking on, and print the reasoning to standard error before the answer. |
| `--max-tokens N` | Limit the answer length. Default: 1024, or 8192 with `--think`. |

Put `message` in single quotes. In zsh, a double-quoted message that ends in `!` leaves the quote
open.

Exits with status 1 when the endpoint is not running or does not answer.

## serve check

```
uv run serve check <preset> [--only NAMES] [--accuracy]
```

| Option | Description |
|---|---|
| `--only NAMES` | Run only the listed checks, separated by commas. |
| `--accuracy` | Add the `gsm8k` check: the first 100 GSM8K test questions, thinking on. |

Writes `runs/<timestamp>-<preset>/check-<time>.json`. [Validation results](validation-results.md)
lists the checks.

## serve status

```
uv run serve status
```

Prints one line per endpoint with the preset, GPU, session, base URL, and three states:

| Field | Values | Meaning |
|---|---|---|
| `api` | `ready`, `unreachable` | Whether `GET /v1/models` answers through the tunnel. |
| `tunnel` | `up`, `down` | Whether the local `ssh` process of the tunnel runs. |
| `log` | `streaming`, `stopped` | Whether the local `colab exec` that copies the server output to the log runs. |

The server can run on the VM while `log` is `stopped`, for example after the computer slept and
the connection dropped. `api` is the state that tells whether the endpoint works.

## serve connect

```
uv run serve connect <preset>
```

Stops the tunnel process, opens a new tunnel on the same local port, and waits up to 60 seconds
for `GET /v1/models` to answer. The server and the VM are not touched. A Colab runtime accepts one
`colab ssh` connection at a time, so the old tunnel is stopped first.

## serve down

```
uv run serve down <preset>
```

Stops the tunnel and the server, releases the VM, and deletes the endpoint's state file. Before it
deletes the state file, `down` checks `colab sessions`. If the VM is still listed, `down` exits with
status 1 and keeps the state file, so you can run it again.

`up`, `connect`, and `down` stop a local process only if its command line names the preset's
session. A process id left in an old state file, for example after a reboot, is not touched.

## Files

| Path | Contents |
|---|---|
| `presets/<name>.yaml` | Preset files. See [Presets](presets.md). |
| `colab_gpus.yaml` | Colab GPUs. See [GPUs](gpus.md). |
| `.env.local` | Local settings and secrets. Not committed. Template: `.env.example`. |
| `runs/<preset>.state.json` | State of a running endpoint: session, port, process ids, and API key. Mode 600. |
| `runs/<timestamp>-<preset>/resolved.json` | Resolved command, environment, install script, recipe commit, and warnings. |
| `runs/<timestamp>-<preset>/launch.py` | The script that ran on the VM. |
| `runs/<timestamp>-<preset>/tunnel.log` | SSH tunnel output. |
| `runs/<timestamp>-<preset>/check-*.json` | `serve check` reports. |
| `cache/recipes/` | Cached recipe documents from recipes.vllm.ai. |
| `~/colab-runs/<session>.log` | Server log. Starts with `nvidia-smi` output, the vLLM version, and the full command. |

## Environment variables

Each variable is read from the environment first and from `.env.local` second.

| Variable | Description |
|---|---|
| `VLLM_API_KEY` | API key the server requires. Default: `vllm`. |
| `HF_TOKEN` | Hugging Face token, passed to the server. Needed for gated models. |
| `COLAB_BIN` | Path to the `colab` CLI when it is not on `PATH`. |

## Exit status

| Status | Meaning |
|---|---|
| 0 | Success. |
| 1 | The preset can't be resolved, the colab CLI is missing or not signed in, the endpoint is not running, or the launch failed. |
| 2 | Invalid command-line arguments. |
